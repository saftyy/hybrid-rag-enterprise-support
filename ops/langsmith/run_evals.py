"""
run_evals.py — Run the Helix eval set through a pipeline config and score it in LangSmith.

    python -m ops.langsmith.run_evals                       # v1 baseline, all 50 queries
    python -m ops.langsmith.run_evals --config v2 --compare-to v1
    python -m ops.langsmith.run_evals --config v2 --limit 5 # quick smoke run

Every run:
  * is a LangSmith *experiment* on the versioned dataset (prefix = config name), with
    metadata: config, git commit, judge model, generator model, dataset version, prompt hash;
  * writes ops/langsmith/results/<config>.json (latest) and a timestamped copy, plus a
    markdown summary — these files are what the production report quotes;
  * prints the checkpoint gates: custom judge >= 0.75 and faithfulness delta >= +0.05.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from ops.config import get_config
from ops.langsmith.setup import (
    configure_tracing, dataset_name_for, ensure_dataset, get_client, load_test_set,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CUSTOM_JUDGE_GATE = 0.75
FAITHFULNESS_DELTA_GATE = 0.05
SCORE_KEYS = ["faithfulness", "answer_relevance", "csm_resolution_quality",
              "source_hit", "source_recall", "abstained", "citation_valid"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def prompt_hash() -> str:
    import importlib
    from src.generate import SYSTEM_PROMPT
    # importlib, because the package __init__ re-exports functions with the module names
    mods = [importlib.import_module(f"ops.langsmith.evaluators.{m}")
            for m in ("faithfulness", "answer_relevance", "custom_judge")]
    blob = "\n".join([SYSTEM_PROMPT] + [m.SYSTEM for m in mods])
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(mean(vals), 4) if vals else None


def _pct(vals: list[float], p: float) -> float | None:
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    return round(vals[min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))], 1)


def aggregate(rows: list[dict]) -> dict:
    overall = {k: _mean([r["scores"].get(k) for r in rows]) for k in SCORE_KEYS}
    by_difficulty = {}
    for d in sorted({r["difficulty"] for r in rows}):
        sub = [r for r in rows if r["difficulty"] == d]
        by_difficulty[d] = {"n": len(sub),
                            **{k: _mean([r["scores"].get(k) for r in sub]) for k in SCORE_KEYS[:3]}}
    ret = [r["retrieval_latency_ms"] for r in rows]
    gen = [r["generation_latency_ms"] for r in rows]
    latency = {"retrieval_p50_ms": _pct(ret, 50), "retrieval_p95_ms": _pct(ret, 95),
               "generation_p50_ms": _pct(gen, 50), "generation_p95_ms": _pct(gen, 95)}
    return {"n": len(rows), "overall": overall, "by_difficulty": by_difficulty, "latency": latency}


def row_from_result(res: dict) -> dict:
    run, example = res["run"], res["example"]
    out = run.outputs or {}
    scores, comments = {}, {}
    for er in res["evaluation_results"]["results"]:
        scores[er.key] = er.score
        if er.comment:
            comments[er.key] = er.comment
    md = example.metadata or {}
    return {
        "qid": md.get("qid") or example.inputs.get("qid"),
        "difficulty": md.get("difficulty"),
        "category": md.get("category"),
        "question": example.inputs["question"],
        "answer": out.get("answer"),
        "confidence": out.get("confidence"),
        "sources": out.get("sources"),
        "retrieved_doc_ids": out.get("retrieved_doc_ids"),
        "expected_sources": (example.outputs or {}).get("expected_sources"),
        "retrieval_latency_ms": out.get("retrieval_latency_ms"),
        "generation_latency_ms": out.get("generation_latency_ms"),
        "scores": scores,
        "comments": comments,
        "error": str(run.error) if run.error else None,
    }


def gates(summary: dict, baseline: dict | None) -> dict:
    o = summary["overall"]
    g = {"custom_judge_gte_0.75": (o["csm_resolution_quality"] or 0) >= CUSTOM_JUDGE_GATE}
    if baseline:
        delta = (o["faithfulness"] or 0) - (baseline["overall"]["faithfulness"] or 0)
        g["faithfulness_delta"] = round(delta, 4)
        g["faithfulness_delta_gte_0.05"] = delta >= FAITHFULNESS_DELTA_GATE
    return g


def write_outputs(config_name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    latest = RESULTS_DIR / f"{config_name}.json"
    for p in (latest, RESULTS_DIR / f"{config_name}_{stamp}.json"):
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (RESULTS_DIR / f"{config_name}_summary.md").write_text(render_markdown(payload), encoding="utf-8")
    return latest


def render_markdown(p: dict) -> str:
    s, m = p["summary"], p["metadata"]
    lines = [f"# Eval results — config `{m['config']}`", "",
             f"Experiment: `{p.get('experiment_name')}` · dataset `{m['dataset']}` · "
             f"commit `{m['git_commit']}` · judge `{m['judge_model']}` · n={s['n']}", "",
             "| Metric | Score |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" if v is not None else f"| {k} | n/a |"
              for k, v in s["overall"].items()]
    lines += ["", "| Latency | ms |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["latency"].items()]
    lines += ["", "| Difficulty | n | faithfulness | relevance | csm_quality |", "|---|---|---|---|---|"]
    for d, v in s["by_difficulty"].items():
        f = lambda x: f"{x:.3f}" if x is not None else "n/a"  # noqa: E731
        lines.append(f"| {d} | {v['n']} | {f(v['faithfulness'])} | {f(v['answer_relevance'])} "
                     f"| {f(v['csm_resolution_quality'])} |")
    if p.get("gates"):
        lines += ["", "## Checkpoint gates", ""] + [f"- **{k}**: {v}" for k, v in p["gates"].items()]
    lines += ["", "## Lowest CSM-quality answers", "",
              "| QID | csm | faith | rel | hit | comment |", "|---|---|---|---|---|---|"]
    worst = sorted(p["rows"], key=lambda r: r["scores"].get("csm_resolution_quality") or 0)[:10]
    for r in worst:
        sc = r["scores"]
        c = (r["comments"].get("csm_resolution_quality") or "").replace("|", "/")[:160]
        lines.append(f"| {r['qid']} | {sc.get('csm_resolution_quality')} | "
                     f"{sc.get('faithfulness')} | {sc.get('answer_relevance')} | "
                     f"{sc.get('source_hit')} | {c} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="v1")
    ap.add_argument("--limit", type=int, default=None, help="first N queries only (smoke test)")
    ap.add_argument("--compare-to", default=None, help="config whose latest results are the baseline")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--strict", action="store_true", help="exit 1 if a checkpoint gate fails")
    args = ap.parse_args(argv)

    from langsmith.evaluation import evaluate
    from ops.langsmith.evaluators import ALL_EVALUATORS
    from ops.langsmith.evaluators._judge import judge_model_name
    from ops.rag import answer_query, get_retriever

    cfg = get_config(args.config)
    project = configure_tracing()
    client = get_client()
    data = load_test_set()
    ds = ensure_dataset(client, data)

    examples = sorted(client.list_examples(dataset_id=ds.id),
                      key=lambda e: (e.metadata or {}).get("qid", ""))
    if args.limit:
        examples = examples[: args.limit]

    get_retriever(cfg.name)  # build BM25 + clients once, before worker threads start

    def target(inputs: dict) -> dict:
        return answer_query(inputs["question"], cfg.name)

    metadata = {
        "config": cfg.name, **{f"cfg_{k}": v for k, v in cfg.to_metadata().items()},
        "git_commit": git_commit(), "judge_model": judge_model_name(),
        "generation_model": os.environ.get("GENERATION_MODEL", "gpt-4o-mini"),
        "dataset": dataset_name_for(data), "prompt_hash": prompt_hash(),
        "limit": args.limit,
    }
    print(f"Running {len(examples)} examples | config={cfg.name} | project={project}")
    results = evaluate(target, data=examples, evaluators=ALL_EVALUATORS,
                       experiment_prefix=f"{cfg.name}", metadata=metadata,
                       max_concurrency=args.concurrency, client=client)

    rows = sorted((row_from_result(r) for r in results), key=lambda r: r["qid"] or "")
    summary = aggregate(rows)
    baseline = None
    if args.compare_to:
        bpath = RESULTS_DIR / f"{args.compare_to}.json"
        if bpath.exists():
            baseline = json.loads(bpath.read_text(encoding="utf-8"))["summary"]
        else:
            print(f"[warn] no baseline results at {bpath}; run --config {args.compare_to} first")

    payload = {"experiment_name": results.experiment_name, "metadata": metadata,
               "summary": summary, "gates": gates(summary, baseline), "rows": rows}
    path = write_outputs(cfg.name, payload)

    print("\n=== Overall ===")
    for k, v in summary["overall"].items():
        b = baseline["overall"].get(k) if baseline else None
        delta = f"  (Δ {v - b:+.3f} vs {args.compare_to})" if (b is not None and v is not None) else ""
        print(f"  {k:<24} {v if v is None else f'{v:.3f}'}{delta}")
    print("=== Gates ===")
    for k, v in payload["gates"].items():
        print(f"  {k:<30} {v}")
    print(f"\nExperiment: {results.experiment_name}  (LangSmith > Datasets > {ds.name})")
    print(f"Results   : {path}")

    failed = [k for k, v in payload["gates"].items() if v is False]
    if args.strict and failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
