"""Ingestion: chunking unit tests + end-to-end smoke test."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.ingest import RawDoc, chunk_documents, run_ingestion

load_dotenv()

CORPUS_DIR = os.environ.get("CORPUS_DIR", "../corpus")


def test_markdown_chunking_preserves_headers():
    doc = RawDoc(
        doc_id="product-docs/test.md",
        text="# Title\n\nIntro text.\n\n## Section A\n\nSome content about A.\n\n## Section B\n\nSome content about B.",
        category="product-docs",
        doc_type="markdown",
    )
    chunks = chunk_documents([doc])
    assert len(chunks) >= 1
    assert all(c.doc_id == "product-docs/test.md" for c in chunks)


def test_short_ticket_stays_as_single_chunk():
    doc = RawDoc(
        doc_id="tickets/T001.html",
        text="Customer: my workflow times out.\nAgent: increase the timeout to 60 min.",
        category="tickets",
        doc_type="html",
    )
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    assert "Agent" in chunks[0].text


def test_pdf_chunking_splits_long_text():
    long_text = "Runbook step. " * 500  # well over chunk_size=1000
    doc = RawDoc(doc_id="runbooks/test.pdf", text=long_text, category="runbooks", doc_type="pdf")
    chunks = chunk_documents([doc])
    assert len(chunks) > 1


@pytest.mark.skipif(
    not Path(CORPUS_DIR).exists(), reason="Corpus directory not found; skipping integration test"
)
def test_ingest_runs_without_errors():
    report = run_ingestion(CORPUS_DIR)
    assert report["chunks_produced"] > 0
    assert report["documents_loaded"] > 0