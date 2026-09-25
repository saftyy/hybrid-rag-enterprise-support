"""
ingest.py — Validated, traced ingestion for a pipeline config.

    python -m ops.ingest --corpus "../corpus" --config v2

    load (src loaders) -> chunk (src chunker) -> post-process (merge / orphan fold / redact)
      -> embed -> Great Expectations gate -> upsert to Pinecone namespace -> write chunks file

The GE gate sits *between embedding and upsert*: if any expectation fails, nothing is
written to Pinecone and the process exits 1 (so CI or a scheduled job fails loudly). The
previous good index stays live. --force overrides for debugging only.

v1 is the Module 1 index and is not re-ingested here (use `python -m src.ingest`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from langsmith import traceable

from ops.config import PipelineConfig, get_config
from ops.data_quality.run_validations import validate_and_report
from ops.data_quality.transforms import merge_small_chunks, redact_chunks

load_dotenv()


@traceable(name="load_and_chunk")
def load_and_chunk(corpus_dir: str) -> dict:
    from dataclasses import asdict
    from pathlib import Path
    from src.ingest import chunk_documents, load_html_docs, load_markdown_docs, load_pdf_docs

    root = Path(corpus_dir)
    docs = load_markdown_docs(root)
    pdf_docs, skipped = load_pdf_docs(root)
    docs += pdf_docs + load_html_docs(root)
    chunks = [asdict(c) for c in chunk_documents(docs)]
    return {"chunks": chunks, "documents_loaded": len(docs), "skipped_scanned_pdfs": skipped}


@traceable(name="post_process")
def post_process(chunks: list[dict], cfg: PipelineConfig) -> dict:
    before = len(chunks)
    chunks = merge_small_chunks(chunks, cfg.merge_min_chars, cfg.merge_max_chars)
    redacted = 0
    if cfg.redact_pii:
        chunks, redacted = redact_chunks(chunks)
    return {"chunks": chunks, "chunks_before": before, "chunks_after": len(chunks),
            "chunks_redacted": redacted}


def embed(chunks: list[dict]) -> list[list[float]]:
    from langchain_openai import OpenAIEmbeddings
    emb = OpenAIEmbeddings(model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))
    return emb.embed_documents([c["text"] for c in chunks])


@traceable(name="embed")
def traced_embed(chunks: list[dict]) -> dict:
    # Embeddings are returned via a side channel so 1536-float vectors don't bloat the trace.
    vectors = embed(chunks)
    traced_embed.last_vectors = vectors  # type: ignore[attr-defined]
    return {"n_vectors": len(vectors), "dims": sorted({len(v) for v in vectors})}


@traceable(name="upsert")
def upsert(chunks: list[dict], vectors: list[list[float]], cfg: PipelineConfig,
           batch_size: int = 100) -> dict:
    from pinecone import Pinecone
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(cfg.index_name)
    if cfg.namespace:
        try:  # full refresh of this namespace so stale chunk ids don't linger
            index.delete(delete_all=True, namespace=cfg.namespace)
        except Exception:
            pass  # namespace doesn't exist yet
    for i in range(0, len(chunks), batch_size):
        batch = [
            {"id": c["id"], "values": v,
             "metadata": {k: c[k] for k in
                          ("text", "doc_id", "source", "category", "doc_type", "chunk_index")}}
            for c, v in zip(chunks[i:i + batch_size], vectors[i:i + batch_size])
        ]
        index.upsert(vectors=batch, namespace=cfg.namespace)
    return {"upserted": len(chunks), "index": cfg.index_name, "namespace": cfg.namespace}


@traceable(name="ingestion", run_type="chain")
def run(corpus_dir: str, config_name: str, force: bool = False) -> dict:
    cfg = get_config(config_name)
    if cfg.name == "v1":
        raise SystemExit("v1 is the frozen Module 1 index. Use `python -m src.ingest` for it.")

    loaded = load_and_chunk(corpus_dir)
    processed = post_process(loaded["chunks"], cfg)
    chunks = processed["chunks"]
    traced_embed(chunks)
    vectors = traced_embed.last_vectors  # type: ignore[attr-defined]

    ok, report_path = validate_and_report(chunks, vectors, cfg.name, stage="ingestion")
    summary = {"config": cfg.name, "documents_loaded": loaded["documents_loaded"],
               "skipped_scanned_pdfs": loaded["skipped_scanned_pdfs"],
               **{k: v for k, v in processed.items() if k != "chunks"},
               "validation_passed": ok, "validation_report": str(report_path)}
    if not ok and not force:
        print(f"\n[BLOCKED] Data-quality gate failed — nothing upserted. See {report_path}")
        summary["upserted"] = 0
        return summary

    summary.update(upsert(chunks, vectors, cfg))
    cfg.chunks_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.chunks_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--config", default="v2")
    ap.add_argument("--force", action="store_true", help="upsert even if validation fails")
    a = ap.parse_args()
    s = run(a.corpus, a.config, a.force)
    print(json.dumps(s, indent=2))
    if not s["validation_passed"] and not a.force:
        sys.exit(1)


if __name__ == "__main__":
    main()
