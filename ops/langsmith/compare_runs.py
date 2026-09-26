"""
compare_runs.py — Per-question diff of one metric between two saved eval runs.

    python -m ops.langsmith.compare_runs v1 v2-vector
    python -m ops.langsmith.compare_runs v2-vector v3 --metric csm_resolution_quality
    python -m ops.langsmith.compare_runs v1 v3 --all       # include unchanged perfect scores

Reads ops/langsmith/results/<config>.json (written by run_evals). Shows every question where
the metric is below 1.0 in either run, plus the judge's comment from the second run, and a
guardrail summary (abstention / relevance / CSM quality) so a faithfulness gain that was
bought by abstaining more is visible immediately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
GUARDRAILS = ["abstained", "answer_relevance", "csm_resolution_quality", "source_hit"]


def load_rows(config: str) -> dict[str, dict]:
    data = json.loads((RESULTS / f"{config}.json").read_text(encoding="utf-8"))
    return {r["qid"]: r for r in data["rows"]}


def fmt(v) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def diff_rows(a: dict, b: dict, metric: str, show_all: bool = False) -> list[tuple]:
    out = []
    for qid in sorted(set(a) | set(b)):
        va = a.get(qid, {}).get("scores", {}).get(metric)
        vb = b.get(qid, {}).get("scores", {}).get(metric)
        # `is not None` on purpose: a score of 0.0 is the most important row, not a falsy one
        below = any(v is not None and v < 1.0 for v in (va, vb))
        if show_all or below or va != vb:
            comment = (b.get(qid, {}).get("comments", {}).get(metric) or "").replace("\n", " ")
            out.append((qid, va, vb, comment))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--metric", default="faithfulness")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--width", type=int, default=220)
    args = ap.parse_args(argv)

    a, b = load_rows(args.base), load_rows(args.new)
    rows = diff_rows(a, b, args.metric, args.all)
    print(f"\n{args.metric}: {args.base} -> {args.new}   ({len(rows)} questions shown)\n")
    for qid, va, vb, comment in rows:
        arrow = "▲" if (va is not None and vb is not None and vb > va) else \
                "▼" if (va is not None and vb is not None and vb < va) else " "
        print(f"{qid}  {fmt(va)} -> {fmt(vb)} {arrow}  | {comment[:args.width]}")

    print("\nGuardrails (mean over questions):")
    for m in GUARDRAILS:
        ma = [r["scores"].get(m) for r in a.values() if r["scores"].get(m) is not None]
        mb = [r["scores"].get(m) for r in b.values() if r["scores"].get(m) is not None]
        if ma and mb:
            print(f"  {m:<24} {sum(ma)/len(ma):.3f} -> {sum(mb)/len(mb):.3f}")

    newly_abstained = [q for q in b if b[q]["scores"].get("abstained") == 1.0
                       and a.get(q, {}).get("scores", {}).get("abstained") == 0.0]
    if newly_abstained:
        print(f"\n  ⚠ newly abstaining in {args.new}: {', '.join(newly_abstained)}")


if __name__ == "__main__":
    main()
