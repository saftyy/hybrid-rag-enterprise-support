"""
calibrate_judge.py — Does the custom judge agree with a human?

A judge you haven't checked against human judgement is just a second model's opinion.
Procedure (≈45 minutes, do it once, after the v1 baseline run):

  1. Open ops/langsmith/calibration/human_labels.csv (15 stratified queries pre-listed).
  2. For each qid, read the question, the reference answer (evals/test_set.json) and the
     *system's* answer (ops/langsmith/results/v1.json -> rows). Do NOT look at the judge's
     score first — that anchors you.
  3. Fill `human_score` with one of 0, 0.25, 0.5, 0.75, 1.0 using the same question the
     judge answers: "Could a senior CSM safely act on this in front of a customer?"
     Add a one-line `human_note`.
  4. Run:  python -m ops.langsmith.calibrate_judge --config v1

Reported: mean absolute error, % within ±0.25, Spearman rank correlation, and pass/fail
agreement at the 0.75 gate with Cohen's kappa. Target: MAE ≤ 0.15 and kappa ≥ 0.6. If the
judge is systematically harsher/softer than you, adjust the rubric wording (not the
threshold), re-run, and record the change — that iteration *is* the calibration story.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
LABELS = HERE / "calibration" / "human_labels.csv"
RESULTS = HERE / "results"
GATE = 0.75


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def calibration_stats(human: list[float], judge: list[float]) -> dict:
    df = pd.DataFrame({"h": human, "j": judge})
    hp, jp = list(df.h >= GATE), list(df.j >= GATE)
    return {
        "n": len(df),
        "mae": round(float((df.h - df.j).abs().mean()), 3),
        "within_0.25": round(float(((df.h - df.j).abs() <= 0.25).mean()), 3),
        "spearman": round(float(df.h.rank().corr(df.j.rank())), 3) if len(df) > 2 else None,
        "gate_agreement": round(sum(x == y for x, y in zip(hp, jp)) / len(df), 3),
        "cohen_kappa": round(cohen_kappa(hp, jp), 3),
        "judge_minus_human_mean": round(float((df.j - df.h).mean()), 3),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="v1")
    args = ap.parse_args(argv)

    res = json.loads((RESULTS / f"{args.config}.json").read_text(encoding="utf-8"))
    judge = {r["qid"]: r["scores"].get("csm_resolution_quality") for r in res["rows"]}
    with open(LABELS, encoding="utf-8") as f:
        labels = [r for r in csv.DictReader(f) if r.get("human_score", "").strip()]
    if not labels:
        raise SystemExit(f"No human scores filled in {LABELS} yet.")

    pairs = [(float(r["human_score"]), judge[r["qid"]], r) for r in labels
             if judge.get(r["qid"]) is not None]
    stats = calibration_stats([p[0] for p in pairs], [p[1] for p in pairs])

    lines = [f"# Custom judge calibration (config `{args.config}`)", "",
             *(f"- **{k}**: {v}" for k, v in stats.items()), "",
             "| QID | human | judge | diff | human note |", "|---|---|---|---|---|"]
    for h, j, r in sorted(pairs, key=lambda p: -abs(p[0] - p[1])):
        lines.append(f"| {r['qid']} | {h} | {j} | {j - h:+.2f} | {r.get('human_note', '')} |")
    out = HERE / "calibration" / "calibration_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
