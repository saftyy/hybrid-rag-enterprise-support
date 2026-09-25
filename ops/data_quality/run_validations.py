"""
run_validations.py — Run the Great Expectations chunk contract and write a report.

    python -m ops.data_quality.run_validations                 # validates v2 chunks
    python -m ops.data_quality.run_validations --config v1     # the Module 1 chunks (fails: good demo)
    python -m ops.data_quality.run_validations --no-embeddings # offline, skip dim checks

Embeddings: by default the stored vectors are fetched back from Pinecone for every chunk
id, so the dimension check validates what is *actually in the index*, not what we think
we wrote. The same suite runs inside ops.ingest before upsert (the blocking gate).

Output: ops/data_quality/results/validation_<config>.md (+ .json, + timestamped copy).
Exit code 1 if any expectation fails — that is what makes it a CI gate.
The report never contains chunk text (it may contain PII); only chunk ids and counts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from ops.config import get_config
from ops.data_quality.expectations import (
    MIN_CHUNK_CHARS, HARD_MIN_CHARS, HARD_MAX_CHARS, chunks_to_dataframe, run_suite,
)
from ops.data_quality.transforms import find_pii

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def offenders(chunks: list[dict]) -> dict:
    """Chunk ids behind the most actionable failures (ids only — never text)."""
    short = sorted((len(c["text"]), c["id"]) for c in chunks if len(c["text"]) < MIN_CHUNK_CHARS)
    hard = [c["id"] for c in chunks
            if not HARD_MIN_CHARS <= len(c["text"]) <= HARD_MAX_CHARS]
    pii = {c["id"]: find_pii(c["text"]) for c in chunks if find_pii(c["text"])}
    return {"short_chunks": [f"{cid} ({n} chars)" for n, cid in short[:15]],
            "short_chunk_count": len(short), "hard_bound_violations": hard, "pii_chunks": pii}


def fetch_embeddings_from_pinecone(chunk_ids: list[str], namespace: str) -> list[list[float] | None]:
    from pinecone import Pinecone
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
        os.environ.get("PINECONE_INDEX_NAME", "helix-checkpoint-rag"))
    found: dict[str, list[float]] = {}
    for i in range(0, len(chunk_ids), 100):
        resp = index.fetch(ids=chunk_ids[i:i + 100], namespace=namespace)
        vectors = resp["vectors"] if isinstance(resp, dict) else resp.vectors
        for vid, v in vectors.items():
            found[vid] = list(v["values"] if isinstance(v, dict) else v.values)
    return [found.get(cid) for cid in chunk_ids]   # None -> missing from index (fails check)


def render_report(config: str, stage: str, results: list[dict], offs: dict, n: int) -> str:
    passed = sum(r["success"] for r in results)
    status = "PASS" if passed == len(results) else "FAIL"
    lines = [f"# Data quality report — `{config}` ({stage})", "",
             f"**{status}** — {passed}/{len(results)} expectations passed on {n} chunks · "
             f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}", "",
             "| Group | Expectation | Column | Result | Observed / unexpected |",
             "|---|---|---|---|---|"]
    for r in results:
        obs = r["observed_value"]
        if r["unexpected_count"] is not None:
            obs = f"{r['unexpected_count']} rows ({(r['unexpected_percent'] or 0):.1f}%)"
        if r["exception"]:
            obs = f"ERROR: {r['exception']}"
        lines.append(f"| {r['group']} | `{r['expectation']}` | {r['column']} | "
                     f"{'✅' if r['success'] else '❌'} | {obs} |")
    lines += ["", "## Offending chunks (ids only)", "",
              f"- Chunks under {MIN_CHUNK_CHARS} chars: **{offs['short_chunk_count']}**"
              + (f" — e.g. {', '.join(offs['short_chunks'][:8])}" if offs["short_chunks"] else ""),
              f"- Outside hard bounds [{HARD_MIN_CHARS}, {HARD_MAX_CHARS}]: "
              f"{offs['hard_bound_violations'] or 'none'}",
              f"- Chunks with PII matches: "
              + (", ".join(f"{k} {v}" for k, v in offs["pii_chunks"].items()) or "none")]
    return "\n".join(lines) + "\n"


def validate_and_report(chunks: list[dict], embeddings: list | None, config: str,
                        stage: str = "standalone") -> tuple[bool, Path]:
    df = chunks_to_dataframe(chunks, embeddings)
    results = run_suite(df, include_embeddings=embeddings is not None)
    ok = all(r["success"] for r in results)
    offs = offenders(chunks)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md = render_report(config, stage, results, offs, len(chunks))
    payload = {"config": config, "stage": stage, "passed": ok, "n_chunks": len(chunks),
               "embeddings_checked": embeddings is not None, "results": results,
               "offenders": offs}
    latest = RESULTS_DIR / f"validation_{config}.md"
    latest.write_text(md, encoding="utf-8")
    (RESULTS_DIR / f"validation_{config}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (RESULTS_DIR / f"validation_{config}_{stamp}.md").write_text(md, encoding="utf-8")
    return ok, latest


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="v2")
    ap.add_argument("--no-embeddings", action="store_true")
    args = ap.parse_args(argv)

    cfg = get_config(args.config)
    if not cfg.chunks_path.exists():
        raise SystemExit(f"{cfg.chunks_path} not found — run ingestion for '{cfg.name}' first.")
    chunks = json.loads(cfg.chunks_path.read_text(encoding="utf-8"))

    embeddings = None
    if not args.no_embeddings:
        if os.environ.get("PINECONE_API_KEY"):
            print(f"Fetching {len(chunks)} stored vectors from Pinecone (namespace "
                  f"'{cfg.namespace or 'default'}') ...")
            embeddings = fetch_embeddings_from_pinecone([c["id"] for c in chunks], cfg.namespace)
        else:
            print("[warn] PINECONE_API_KEY not set — skipping embedding checks.")

    ok, path = validate_and_report(chunks, embeddings, cfg.name)
    print(re.sub(r"\n{3,}", "\n\n", path.read_text(encoding="utf-8")))
    print(f"Report: {path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
