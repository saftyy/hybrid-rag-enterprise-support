# Helix Support Assistant — RAG Checkpoint

> baswe.Ai Engineer Accelerator™ — LLMs & RAG Checkpoint (Part 1: The Build)

## Summary
<!-- TODO: one paragraph — what you built and what it does -->

## Architecture
<!-- TODO: Mermaid diagram of ingestion -> retrieval -> generation -->

## Chunking Strategy
<!-- TODO: chunk size, overlap, how Markdown / PDF / HTML are each handled, and why -->

## Retrieval Strategy
<!-- TODO: which improvement (hybrid / rerank / multi-query), why, before/after metrics -->

## Generation Prompt
<!-- TODO: actual prompt template used, with reasoning for the structure -->

## Eval Results
<!-- TODO: table — faithfulness, answer relevance, context precision + 3 failure cases -->

## How to Run

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
python -m src.ingest --corpus "../YoussefElsafty.AI - LLMs & RAG corpus/corpus"
python -m src.pipeline "your question here"
python -m evals.harness
```

## Loom Walkthrough
<!-- TODO: link -->

## What I'd Do With Another Week
<!-- TODO: ranked list of next improvements with reasoning -->
