"""
ingest.py — Document loader -> chunker -> embedder -> vector store.

Run:
    python -m src.ingest --corpus ./corpus

Handles three formats from the Helix corpus:
    - Markdown (product-docs/)   -> header-aware splitting
    - PDF, text-native (runbooks/) -> recursive character splitting
    - HTML (tickets/)             -> whole-ticket chunks where possible

Scanned PDFs (no text layer) are detected and skipped gracefully — this is
a conscious engineering decision (Option B: skip rather than OCR), documented
in the README. Ingestion never crashes on an unreadable file.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from pypdf import PdfReader
from bs4 import BeautifulSoup
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

import os

load_dotenv()

# Below this many extracted characters, a PDF is treated as scanned/unreadable.
MIN_PDF_TEXT_CHARS = 50

# A ticket HTML file at or under this length is kept as a single chunk, so a
# question and its resolution (often only in the final turn) never get split
# across chunks.
TICKET_SINGLE_CHUNK_MAX_CHARS = 3000

MD_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]


@dataclass
class RawDoc:
    doc_id: str      # path relative to corpus root, e.g. "runbooks/csm/onboarding.pdf"
    text: str
    category: str    # product-docs / runbooks / tickets
    doc_type: str     # markdown / pdf / html


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    source: str
    category: str
    doc_type: str
    chunk_index: int


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def load_markdown_docs(corpus_root: Path) -> list[RawDoc]:
    root = corpus_root / "product-docs"
    docs: list[RawDoc] = []
    for path in sorted(root.rglob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  [WARN] Failed to read {path}: {e}")
            continue
        doc_id = str(path.relative_to(corpus_root)).replace("\\", "/")
        docs.append(RawDoc(doc_id=doc_id, text=text, category="product-docs", doc_type="markdown"))
    return docs


def load_pdf_docs(corpus_root: Path) -> tuple[list[RawDoc], list[str]]:
    root = corpus_root / "runbooks"
    docs: list[RawDoc] = []
    skipped: list[str] = []
    for path in sorted(root.rglob("*.pdf")):
        doc_id = str(path.relative_to(corpus_root)).replace("\\", "/")
        try:
            reader = PdfReader(str(path))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
        except Exception as e:
            print(f"  [WARN] Failed to read {path}: {e}")
            skipped.append(doc_id)
            continue

        if len(text) < MIN_PDF_TEXT_CHARS:
            # No usable text layer -> scanned PDF. Skip gracefully (Option B).
            print(f"  [SKIP] {doc_id} — no extractable text (likely scanned)")
            skipped.append(doc_id)
            continue

        docs.append(RawDoc(doc_id=doc_id, text=text, category="runbooks", doc_type="pdf"))
    return docs, skipped


def load_html_docs(corpus_root: Path) -> list[RawDoc]:
    root = corpus_root / "tickets"
    docs: list[RawDoc] = []
    for path in sorted(root.rglob("*.html")):
        doc_id = str(path.relative_to(corpus_root)).replace("\\", "/")
        try:
            raw = path.read_text(encoding="utf-8")
            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        except Exception as e:
            print(f"  [WARN] Failed to read {path}: {e}")
            continue
        docs.append(RawDoc(doc_id=doc_id, text=text, category="tickets", doc_type="html"))
    return docs


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def chunk_documents(docs: list[RawDoc]) -> list[Chunk]:
    chunks: list[Chunk] = []

    md_header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MD_HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""]
    )
    ticket_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
    )

    for doc in docs:
        if doc.doc_type == "markdown":
            sections = md_header_splitter.split_text(doc.text)
            idx = 0
            for section in sections:
                for sub in recursive_splitter.split_text(section.page_content):
                    chunks.append(_make_chunk(doc, sub, idx))
                    idx += 1

        elif doc.doc_type == "pdf":
            for idx, sub in enumerate(recursive_splitter.split_text(doc.text)):
                chunks.append(_make_chunk(doc, sub, idx))

        elif doc.doc_type == "html":
            if len(doc.text) <= TICKET_SINGLE_CHUNK_MAX_CHARS:
                chunks.append(_make_chunk(doc, doc.text, 0))
            else:
                for idx, sub in enumerate(ticket_splitter.split_text(doc.text)):
                    chunks.append(_make_chunk(doc, sub, idx))

    return chunks


def _make_chunk(doc: RawDoc, text: str, idx: int) -> Chunk:
    return Chunk(
        id=f"{doc.doc_id}::{idx}",
        text=text,
        doc_id=doc.doc_id,
        source=doc.doc_id,
        category=doc.category,
        doc_type=doc.doc_type,
        chunk_index=idx,
    )


# --------------------------------------------------------------------------
# Embedding + indexing
# --------------------------------------------------------------------------

def get_pinecone_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ.get("PINECONE_INDEX_NAME", "helix-checkpoint-rag")
    cloud = os.environ.get("PINECONE_CLOUD", "aws")
    region = os.environ.get("PINECONE_REGION", "us-east-1")

    existing = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}' ({cloud}/{region})...")
        pc.create_index(
            name=index_name,
            dimension=1536,  # text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    return pc.Index(index_name)


def embed_and_upsert(chunks: list[Chunk], batch_size: int = 100) -> None:
    embeddings = OpenAIEmbeddings(model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))
    index = get_pinecone_index()

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding + upserting"):
        batch = chunks[i:i + batch_size]
        vectors = embeddings.embed_documents([c.text for c in batch])
        to_upsert = [
            {
                "id": c.id,
                "values": vec,
                "metadata": {
                    "text": c.text,
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "category": c.category,
                    "doc_type": c.doc_type,
                    "chunk_index": c.chunk_index,
                },
            }
            for c, vec in zip(batch, vectors)
        ]
        index.upsert(vectors=to_upsert)


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------

def run_ingestion(corpus_dir: str) -> dict:
    corpus_root = Path(corpus_dir)

    print(f"Loading documents from {corpus_root} ...")
    docs: list[RawDoc] = []
    docs += load_markdown_docs(corpus_root)
    pdf_docs, skipped_pdfs = load_pdf_docs(corpus_root)
    docs += pdf_docs
    docs += load_html_docs(corpus_root)

    print(f"Loaded {len(docs)} documents ({len(skipped_pdfs)} scanned PDFs skipped).")

    print("Chunking...")
    chunks = chunk_documents(docs)
    print(f"Produced {len(chunks)} chunks.")

    print("Embedding + upserting to Pinecone...")
    embed_and_upsert(chunks)

    # Persist chunks to disk so retrieve.py can build a BM25 index without
    # re-parsing the entire corpus on every run.
    chunks_path = Path(__file__).resolve().parent.parent / "evals" / "results" / "chunks.json"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "id": c.id,
                    "text": c.text,
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "category": c.category,
                    "doc_type": c.doc_type,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
            f,
            indent=2,
        )

    report = {
        "documents_loaded": len(docs),
        "documents_skipped_scanned_pdfs": skipped_pdfs,
        "chunks_produced": len(chunks),
    }
    report_path = Path(__file__).resolve().parent.parent / "evals" / "results" / "ingestion_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Ingestion complete. Report written to {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Ingest the Helix corpus into Pinecone.")
    parser.add_argument("--corpus", required=True, help="Path to the corpus directory")
    args = parser.parse_args()
    run_ingestion(args.corpus)


if __name__ == "__main__":
    main()