"""
instrument.py — Launch Phoenix, trace sample queries through the RAG, summarise the spans.

    python -m ops.phoenix.instrument                  # v1, 15 sample queries, keeps UI open
    python -m ops.phoenix.instrument --config v2
    python -m ops.phoenix.instrument --all --no-wait  # all 50 queries, exit when done

Span tree per query (OpenInference conventions, so Phoenix renders them natively):

    rag_query            CHAIN      input question, output answer, confidence, eval hints
    ├── retrieve         RETRIEVER  doc ids, scores, chunk text; latency; chunk lengths
    └── generate         CHAIN      wraps the LCEL chain; LangChain auto-instrumentation adds
        └── ChatOpenAI   LLM        prompt/completion tokens, model, latency

The OpenTelemetry pipeline is wired by hand (TracerProvider + OTLP/HTTP exporter) rather
than via phoenix.otel.register(): it is 5 lines, and it avoids a version incompatibility
between phoenix-otel and the pinned OpenTelemetry SDK.

After the run, spans are pulled back from Phoenix and summarised into
ops/phoenix/results/phoenix_stats_<config>.json / .md — the numbers analysis.md and the
monitoring spec quote (p50/p95 latency per stage, tokens, retrieved-chunk sizes).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from ops.config import get_config
from ops.langsmith.evaluators import is_abstention

load_dotenv()

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
TEST_SET = HERE.parents[1] / "evals" / "test_set.json"

# Known Module 1 failures (Q017, Q029, Q033, Q037, Q038, Q044, Q046) + a spread of passes.
SAMPLE_QIDS = ["Q001", "Q003", "Q010", "Q017", "Q020", "Q023", "Q029", "Q033",
               "Q037", "Q038", "Q044", "Q045", "Q046", "Q047", "Q050"]

# gpt-4o-mini list price per 1M tokens (USD). Update if you change GENERATION_MODEL.
PRICE_IN, PRICE_OUT = 0.15, 0.60
SHORT_CHUNK = 200


def build_tracer_provider(project: str, endpoint: str):
    from openinference.semconv.resource import ResourceAttributes
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    tp = TracerProvider(resource=Resource({ResourceAttributes.PROJECT_NAME: project}))
    tp.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    return tp


def instrument_langchain(tp) -> None:
    from openinference.instrumentation.langchain import LangChainInstrumentor
    LangChainInstrumentor().instrument(tracer_provider=tp)


def set_retrieval_attributes(span, chunks: list[dict]) -> None:
    """RETRIEVER span attributes: per-document id/score/content + summary stats."""
    from openinference.semconv.trace import DocumentAttributes as D
    from openinference.semconv.trace import SpanAttributes as S

    span.set_attribute(S.OPENINFERENCE_SPAN_KIND, "RETRIEVER")
    for i, c in enumerate(chunks):
        p = f"{S.RETRIEVAL_DOCUMENTS}.{i}."
        span.set_attribute(p + D.DOCUMENT_ID, c["id"])
        span.set_attribute(p + D.DOCUMENT_SCORE, float(c.get("score") or 0.0))
        span.set_attribute(p + D.DOCUMENT_CONTENT, c["text"])
        span.set_attribute(p + D.DOCUMENT_METADATA, json.dumps(
            {"doc_id": c["doc_id"], "category": c.get("category"),
             "vector_rank": c.get("vector_rank"), "bm25_rank": c.get("bm25_rank")}))
    lengths = [len(c["text"]) for c in chunks]
    span.set_attribute("rag.retrieval.k", len(chunks))
    span.set_attribute("rag.retrieval.doc_ids", [c["doc_id"] for c in chunks])
    span.set_attribute("rag.retrieval.distinct_docs", len({c["doc_id"] for c in chunks}))
    span.set_attribute("rag.retrieval.chunk_lengths", lengths)
    span.set_attribute("rag.retrieval.short_chunks", sum(n < SHORT_CHUNK for n in lengths))


def traced_query(tracer, retriever, q: dict, config_name: str) -> dict:
    from openinference.semconv.trace import SpanAttributes as S
    from ops.generation import generate_for

    with tracer.start_as_current_span("rag_query") as root:
        root.set_attribute(S.OPENINFERENCE_SPAN_KIND, "CHAIN")
        root.set_attribute(S.INPUT_VALUE, q["query"])
        root.set_attribute("rag.config", config_name)
        root.set_attribute("eval.qid", q["id"])

        t0 = time.perf_counter()
        with tracer.start_as_current_span("retrieve") as rs:
            rs.set_attribute(S.INPUT_VALUE, q["query"])
            chunks = retriever.retrieve_scored(q["query"])
            set_retrieval_attributes(rs, chunks)
        t1 = time.perf_counter()
        with tracer.start_as_current_span("generate") as gs:
            gs.set_attribute(S.OPENINFERENCE_SPAN_KIND, "CHAIN")
            result = generate_for(get_config(config_name).prompt, q["query"],
                                  [{"doc_id": c["doc_id"], "text": c["text"]} for c in chunks])
            gs.set_attribute(S.OUTPUT_VALUE, result.answer)
        t2 = time.perf_counter()

        doc_ids = [c["doc_id"] for c in chunks]
        hit = bool(set(q.get("expected_sources", [])) & set(doc_ids))
        abst = is_abstention(result.answer)
        root.set_attribute(S.OUTPUT_VALUE, result.answer)
        root.set_attribute("rag.confidence", result.confidence)
        root.set_attribute("eval.source_hit", hit)
        root.set_attribute("eval.abstained", abst)

    return {"qid": q["id"], "question": q["query"], "answer": result.answer,
            "confidence": result.confidence, "retrieved_chunk_ids": [c["id"] for c in chunks],
            "retrieved_doc_ids": doc_ids, "expected_sources": q.get("expected_sources", []),
            "chunk_lengths": [len(c["text"]) for c in chunks], "source_hit": hit,
            "abstained": abst, "retrieval_ms": round((t1 - t0) * 1000, 1),
            "generation_ms": round((t2 - t1) * 1000, 1)}


def pct(vals: list[float], p: float) -> float | None:
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    return round(vals[min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))], 1)


def span_stats(df) -> dict:
    """Summarise a Phoenix spans dataframe."""
    if df is None or len(df) == 0:
        return {}
    dur = (df["end_time"] - df["start_time"]).dt.total_seconds() * 1000
    df = df.assign(duration_ms=dur)
    out = {}
    for label, mask in {"retrieval": df["span_kind"] == "RETRIEVER",
                        "llm": df["span_kind"] == "LLM",
                        "end_to_end": df["name"] == "rag_query"}.items():
        d = list(df.loc[mask, "duration_ms"])
        out[f"{label}_latency_ms"] = {"n": len(d), "p50": pct(d, 50), "p95": pct(d, 95),
                                      "max": round(max(d), 1) if d else None}
    tok = {}
    for col, key in [("attributes.llm.token_count.prompt", "prompt"),
                     ("attributes.llm.token_count.completion", "completion"),
                     ("attributes.llm.token_count.total", "total")]:
        if col in df.columns:
            v = [float(x) for x in df.loc[df["span_kind"] == "LLM", col].dropna()]
            if v:
                tok[key] = {"mean": round(mean(v), 1), "p95": pct(v, 95)}
    if tok:
        out["tokens_per_llm_call"] = tok
        if "prompt" in tok and "completion" in tok:
            out["est_cost_per_query_usd"] = round(
                (tok["prompt"]["mean"] * PRICE_IN + tok["completion"]["mean"] * PRICE_OUT) / 1e6, 6)
    return out


def local_stats(rows: list[dict]) -> dict:
    lengths = [n for r in rows for n in r["chunk_lengths"]]
    return {
        "queries": len(rows),
        "source_hit_rate": round(mean(r["source_hit"] for r in rows), 3),
        "abstention_rate": round(mean(r["abstained"] for r in rows), 3),
        "retrieved_chunks": len(lengths),
        "retrieved_chunk_len_median": sorted(lengths)[len(lengths) // 2] if lengths else None,
        f"retrieved_chunks_under_{SHORT_CHUNK}_pct": round(
            sum(n < SHORT_CHUNK for n in lengths) / len(lengths), 3) if lengths else None,
        "client_retrieval_ms": {"p50": pct([r["retrieval_ms"] for r in rows], 50),
                                "p95": pct([r["retrieval_ms"] for r in rows], 95)},
        "client_generation_ms": {"p50": pct([r["generation_ms"] for r in rows], 50),
                                 "p95": pct([r["generation_ms"] for r in rows], 95)},
    }


def render_md(config: str, stats: dict, rows: list[dict]) -> str:
    lines = [f"# Phoenix run — `{config}`", "", "```json", json.dumps(stats, indent=2), "```", "",
             "| QID | hit | abstained | conf | retrieved chunk lengths | ret ms | gen ms |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['qid']} | {'✅' if r['source_hit'] else '❌'} | "
                     f"{'yes' if r['abstained'] else ''} | {r['confidence']} | "
                     f"{r['chunk_lengths']} | {r['retrieval_ms']} | {r['generation_ms']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="v1")
    ap.add_argument("--all", action="store_true", help="run all 50 test queries")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PHOENIX_PORT", 6006)))
    ap.add_argument("--no-launch", action="store_true", help="use an already-running Phoenix")
    ap.add_argument("--no-wait", action="store_true", help="don't keep the UI open at the end")
    args = ap.parse_args(argv)

    cfg = get_config(args.config)
    project = f"helix-rag-{cfg.name}"
    base_url = f"http://localhost:{args.port}"

    if not args.no_launch:
        os.environ["PHOENIX_PORT"] = str(args.port)
        import phoenix as px
        session = px.launch_app()
        print(f"Phoenix UI: {session.url}")

    tp = build_tracer_provider(project, f"{base_url}/v1/traces")
    instrument_langchain(tp)
    tracer = tp.get_tracer("helix.rag")

    from ops.rag import get_retriever
    retriever = get_retriever(cfg.name)
    queries = json.loads(TEST_SET.read_text(encoding="utf-8"))["queries"]
    if not args.all:
        queries = [q for q in queries if q["id"] in SAMPLE_QIDS]

    rows = []
    for q in queries:
        r = traced_query(tracer, retriever, q, cfg.name)
        rows.append(r)
        print(f"  {r['qid']}  hit={'Y' if r['source_hit'] else 'N'}  "
              f"abstain={'Y' if r['abstained'] else 'N'}  ret={r['retrieval_ms']}ms  "
              f"gen={r['generation_ms']}ms  chunks={r['chunk_lengths']}")
    tp.force_flush()
    time.sleep(2)

    stats = {"config": cfg.name, "project": project, **local_stats(rows)}
    try:
        from phoenix.client import Client
        df = Client(base_url=base_url).spans.get_spans_dataframe(
            project_identifier=project, limit=10000)
        stats["phoenix_spans"] = span_stats(df)
    except Exception as e:  # stats above are still valid without this
        stats["phoenix_spans_error"] = str(e)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"phoenix_stats_{cfg.name}.json").write_text(
        json.dumps({"stats": stats, "rows": rows}, indent=2), encoding="utf-8")
    (RESULTS_DIR / f"phoenix_stats_{cfg.name}.md").write_text(
        render_md(cfg.name, stats, rows), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {RESULTS_DIR / f'phoenix_stats_{cfg.name}.md'}")

    if not args.no_launch and not args.no_wait:
        # (On Windows, a PermissionError about phoenix.db may print after you press Enter:
        #  Phoenix's temp database is still locked during interpreter shutdown. Harmless.)
        input(f"\nPhoenix is running at {base_url} (project '{project}'). Take your screenshots "
              "into ops/phoenix/screenshots/, then press Enter to exit...")


if __name__ == "__main__":
    main()
