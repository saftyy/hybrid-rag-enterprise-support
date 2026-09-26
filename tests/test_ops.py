"""
test_ops.py — Tests for the Module 3 operational layer.

All tests here are OFFLINE: no OpenAI / Pinecone / LangSmith / Phoenix calls. LLM judges
get a fake `judge` callable, Pinecone/embeddings get fakes, OpenTelemetry uses an
in-memory exporter. They run in a few seconds and cost nothing, so they can run on every
commit even without secrets.
"""

import os

# Tests must never ship traces to LangSmith. Set before any ops import (langsmith caches
# env lookups), and before load_dotenv() runs (it does not override existing vars).
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from ops.config import CONFIGS, REPO_ROOT, get_config  # noqa: E402
from ops.data_quality import expectations as E  # noqa: E402
from ops.data_quality.transforms import (  # noqa: E402
    ORPHAN_CHARS, find_pii, merge_small_chunks, redact_chunks, redact_pii,
)
from ops.langsmith.evaluators import (  # noqa: E402
    abstained, citation_valid, is_abstention, source_hit, source_recall,
)
from ops.langsmith.evaluators.answer_relevance import (  # noqa: E402
    RelevanceVerdict, score_answer_relevance,
)
from ops.langsmith.evaluators.custom_judge import (  # noqa: E402
    COMMITMENT_CAP, CSMVerdict, combine, score_csm_resolution_quality,
)
from ops.langsmith.evaluators.faithfulness import (  # noqa: E402
    ClaimCheck, FaithfulnessVerdict, score_faithfulness,
)


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

def fake_judge(verdict):
    """A JudgeFn that ignores the prompt and returns a fixed verdict."""
    calls = []

    def judge(schema, system, user):
        calls.append((schema, system, user))
        assert isinstance(verdict, schema)
        return verdict

    judge.calls = calls
    return judge


def chunk(doc_id, idx, text, doc_type="markdown", category="product-docs"):
    return {"id": f"{doc_id}::{idx}", "text": text, "doc_id": doc_id, "source": doc_id,
            "category": category, "doc_type": doc_type, "chunk_index": idx}


def clean_chunks(n_docs=95):
    return [chunk(f"product-docs/area/doc{i}.md", 0,
                  f"# Doc {i}\n" + "Helix workflows support retries and timeouts. " * 12)
            for i in range(n_docs)]


# ---------------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------------

def test_v1_config_matches_module1_baseline():
    v1 = get_config("v1")
    assert v1.retrieval_mode == "hybrid" and v1.k == 5
    assert v1.namespace == "" and v1.merge_min_chars == 0 and not v1.redact_pii
    assert v1.chunks_path == REPO_ROOT / "evals" / "results" / "chunks.json"


def test_v2_configs_never_touch_the_baseline_namespace():
    for name, cfg in CONFIGS.items():
        if name != "v1":
            assert cfg.namespace and cfg.namespace != CONFIGS["v1"].namespace


def test_unknown_config_exits():
    with pytest.raises(SystemExit):
        get_config("v99")


# ---------------------------------------------------------------------------------------
# retriever — v1 must reproduce Module 1 exactly
# ---------------------------------------------------------------------------------------

class FakeEmbeddings:
    def embed_query(self, q):
        return [0.0]


class FakeIndex:
    def __init__(self, ids):
        self.ids, self.last_kwargs = ids, None

    def query(self, **kw):
        self.last_kwargs = kw
        return {"matches": [{"id": i, "score": 1.0 - n * 0.01}
                            for n, i in enumerate(self.ids[: kw["top_k"]])]}


def _retrievers(mode="hybrid", namespace=""):
    from src.retrieve import HybridRetriever
    from ops.rag import ConfigurableRetriever

    chunks = [chunk(f"product-docs/d{i}.md", 0, f"text about topic{i} " + ("webhook url " if i % 3 == 0 else ""))
              for i in range(30)]
    vector_order = [c["id"] for c in reversed(chunks)]
    cfg = get_config("v1").__class__(name="t", description="", chunks_path=REPO_ROOT,
                                     namespace=namespace, retrieval_mode=mode)
    ours = ConfigurableRetriever.__new__(ConfigurableRetriever)
    ours._setup(cfg, chunks, FakeEmbeddings(), FakeIndex(vector_order))

    original = HybridRetriever.__new__(HybridRetriever)
    original.default_k, original.chunks = 5, chunks
    original.chunk_by_id, original.bm25 = ours.chunk_by_id, ours.bm25
    original.embeddings, original.index = FakeEmbeddings(), FakeIndex(vector_order)
    return ours, original, vector_order


def test_v1_hybrid_retrieval_is_identical_to_module1():
    ours, original, _ = _retrievers("hybrid")
    q = "how do I send a webhook to a custom url"
    assert [c["id"] for c in ours.retrieve(q, k=5)] == [c["id"] for c in original.retrieve(q, k=5)]


def test_vector_mode_follows_vector_ranking_and_uses_namespace():
    ours, _, vector_order = _retrievers("vector", namespace="v2")
    got = ours.retrieve_scored("anything", k=5)
    assert [c["id"] for c in got] == vector_order[:5]
    assert all("score" in c for c in got)
    assert ours.index.last_kwargs["namespace"] == "v2"


# ---------------------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------------------

def test_merge_small_markdown_chunks_reaches_min_size():
    doc = "product-docs/api/api-versioning.md"
    chunks = [chunk(doc, 0, "# API Versioning\nVersioned in the URL."),
              chunk(doc, 1, "## Current version\n**v2** is the current stable version."),
              chunk(doc, 2, "## Deprecation\n" + "Old versions are supported for 12 months. " * 10)]
    merged = merge_small_chunks(chunks, min_chars=400, max_chars=1500)
    assert len(merged) < len(chunks)
    assert "v2** is the current stable version" in merged[0]["text"]
    assert [c["id"] for c in merged] == [f"{doc}::{i}" for i in range(len(merged))]


def test_merge_never_crosses_documents_or_exceeds_max():
    a = [chunk("product-docs/a.md", i, "x" * 50) for i in range(3)]
    b = [chunk("product-docs/b.md", i, "y" * 900) for i in range(3)]
    merged = merge_small_chunks(a + b, min_chars=400, max_chars=1000)
    assert all(set(c["text"].replace("\n", "")) <= {"x"} for c in merged if c["doc_id"].endswith("a.md"))
    assert all(len(c["text"]) <= 1000 for c in merged)
    assert sum(c["doc_id"].endswith("b.md") for c in merged) == 3


def test_pdf_orphan_fragment_is_folded_into_previous_chunk():
    doc = "runbooks/incident-response/database-failover.pdf"
    chunks = [chunk(doc, 0, "Restore from the most recent backup", "pdf", "runbooks"),
              chunk(doc, 1, "(RPO is 5 minutes).", "pdf", "runbooks")]
    assert len(chunks[1]["text"]) < ORPHAN_CHARS
    merged = merge_small_chunks(chunks, min_chars=400)
    assert len(merged) == 1 and merged[0]["text"].endswith("(RPO is 5 minutes).")


def test_merge_disabled_is_a_noop():
    chunks = clean_chunks(3)
    assert merge_small_chunks(chunks, min_chars=0) == chunks


def test_redaction_removes_customer_pii_but_keeps_company_addresses():
    text = ("Customer tom@bluefinbrewing.com called from (415) 555-0199. "
            "Escalate to billing-leads@helix.io or support@helix.io.")
    out = redact_pii(text)
    assert "tom@bluefinbrewing.com" not in out and "555-0199" not in out
    assert "billing-leads@helix.io" in out and "support@helix.io" in out
    assert find_pii(out) == {}


def test_redact_chunks_counts_changes():
    chunks = [chunk("tickets/t1.html", 0, "mail erin@skylane.com", "html", "tickets"),
              chunk("tickets/t2.html", 0, "no pii here", "html", "tickets")]
    out, changed = redact_chunks(chunks)
    assert changed == 1 and "[REDACTED_EMAIL]" in out[0]["text"]


# ---------------------------------------------------------------------------------------
# Great Expectations suite
# ---------------------------------------------------------------------------------------

def _run(chunks, dims=E.EXPECTED_EMBEDDING_DIM, with_emb=True):
    emb = [[0.1] * dims for _ in chunks] if with_emb else None
    return E.run_suite(E.chunks_to_dataframe(chunks, emb), include_embeddings=with_emb)


def _failed(results):
    return {(r["group"], r["expectation"], r["column"]) for r in results if not r["success"]}


def test_ge_clean_chunks_pass_every_expectation():
    results = _run(clean_chunks())
    assert results and all(r["success"] for r in results), _failed(results)


def test_ge_catches_fragmented_chunks():
    chunks = clean_chunks()
    for c in chunks[:30]:
        c["text"] = "## Heading\nOne line."
    assert any(g == "chunk_length" for g, _, _ in _failed(_run(chunks)))


def test_ge_catches_pii():
    chunks = clean_chunks()
    chunks[0]["text"] += " Contact jane.doe@customer-corp.com"
    assert ("pii", "expect_column_values_to_not_match_regex_list", "text") in _failed(_run(chunks))


def test_ge_catches_wrong_embedding_dimension():
    failed = _failed(_run(clean_chunks(), dims=3072))
    assert ("embeddings", "expect_column_values_to_be_in_set", "embedding_dim") in failed


def test_ge_catches_missing_metadata_and_dropped_documents():
    chunks = clean_chunks(50)           # < MIN_DISTINCT_DOCS
    chunks[0]["category"] = None
    failed = _failed(_run(chunks))
    assert ("metadata", "expect_column_values_to_not_be_null", "category") in failed
    assert ("metadata", "expect_column_unique_value_count_to_be_between", "doc_id") in failed


def test_ge_embedding_checks_are_skipped_without_embeddings():
    results = _run(clean_chunks(), with_emb=False)
    assert not any(r["group"] == "embeddings" for r in results)


def test_validation_report_is_written_and_contains_no_chunk_text(tmp_path, monkeypatch):
    from ops.data_quality import run_validations as RV
    monkeypatch.setattr(RV, "RESULTS_DIR", tmp_path)
    chunks = clean_chunks()
    chunks[0]["text"] += " secret-customer@acme-corp.com"
    ok, path = RV.validate_and_report(chunks, None, "test")
    assert not ok and path.exists()
    body = path.read_text(encoding="utf-8")
    assert "FAIL" in body and "secret-customer@acme-corp.com" not in body
    assert json.loads((tmp_path / "validation_test.json").read_text())["passed"] is False


def test_ingestion_gate_blocks_upsert_when_validation_fails(monkeypatch):
    from ops import ingest as I
    bad = clean_chunks()
    monkeypatch.setattr(I, "load_and_chunk", lambda d: {"chunks": bad, "documents_loaded": 95,
                                                        "skipped_scanned_pdfs": []})
    monkeypatch.setattr(I, "embed", lambda chunks: [[0.1] * 3072 for _ in chunks])  # wrong dims
    monkeypatch.setattr(I, "validate_and_report",
                        lambda c, v, cfg, stage: (len(v[0]) == 1536, "report.md"))
    upserts = []
    monkeypatch.setattr(I, "upsert", lambda *a, **k: upserts.append(a) or {})
    summary = I.run("unused", "v2")
    assert summary["validation_passed"] is False and summary["upserted"] == 0
    assert upserts == []


# ---------------------------------------------------------------------------------------
# evaluators
# ---------------------------------------------------------------------------------------

def test_faithfulness_is_fraction_of_supported_claims():
    v = FaithfulnessVerdict(claims=[
        ClaimCheck(claim="max 100 steps", supported=True, reason="ctx 1"),
        ClaimCheck(claim="timeout 30 min", supported=True, reason="ctx 1"),
        ClaimCheck(claim="costs $99", supported=False, reason="not in context")])
    out = score_faithfulness("answer", ["ctx"], judge=fake_judge(v))
    assert out["key"] == "faithfulness" and out["score"] == pytest.approx(2 / 3, abs=1e-4)
    assert "costs $99" in out["comment"]


def test_faithfulness_abstention_scores_one():
    out = score_faithfulness("The provided context doesn't cover this.", [],
                             judge=fake_judge(FaithfulnessVerdict(claims=[])))
    assert out["score"] == 1.0


def test_answer_relevance_normalises_1_to_5():
    for raw, expected in [(1, 0.0), (3, 0.5), (5, 1.0)]:
        out = score_answer_relevance("q", "a", judge=fake_judge(RelevanceVerdict(reasoning="r", score=raw)))
        assert out["score"] == expected


@pytest.mark.parametrize("facts,action,commit,expected", [
    ("all", "not_applicable", False, 1.0),
    ("all", "high", False, 1.0),
    ("all", "low", False, 0.76),
    ("partial", "high", False, 0.65),
    ("contradicted", "high", False, 0.3),
    ("all", "high", True, COMMITMENT_CAP),
])
def test_custom_judge_scoring_rules(facts, action, commit, expected):
    v = CSMVerdict(reasoning="r", key_facts=facts, actionability=action, unauthorised_commitment=commit)
    assert combine(v) == pytest.approx(expected)


def test_custom_judge_sends_reference_and_notes_to_the_judge():
    judge = fake_judge(CSMVerdict(reasoning="r", key_facts="all", actionability="high",
                                  unauthorised_commitment=False))
    out = score_csm_resolution_quality("q?", "ans", "REF-100", "must include 100", judge=judge)
    assert out["key"] == "csm_resolution_quality" and out["score"] == 1.0
    user_prompt = judge.calls[0][2]
    assert "REF-100" in user_prompt and "must include 100" in user_prompt


def test_deterministic_evaluators():
    ref = {"expected_sources": ["a.md", "b.md"]}
    out = {"answer": "x", "sources": ["a.md", "z.md"], "retrieved_doc_ids": ["a.md", "c.md"]}
    assert source_hit({}, out, ref)["score"] == 1.0
    assert source_recall({}, out, ref)["score"] == 0.5
    assert citation_valid({}, out, ref)["score"] == 0.0
    assert abstained({}, {"answer": "The provided context doesn't cover this."}, ref)["score"] == 1.0
    assert not is_abstention("The Pro plan costs $299 per month.")


# ---------------------------------------------------------------------------------------
# LangSmith dataset + run_evals plumbing
# ---------------------------------------------------------------------------------------

def test_dataset_examples_and_versioning():
    from ops.langsmith.setup import dataset_name_for, load_test_set, test_set_hash, to_examples
    data = load_test_set()
    ex = to_examples(data)
    assert len(ex) == len(data["queries"]) == 50
    assert set(ex[0]["outputs"]) == {"reference_answer", "expected_sources", "notes"}
    assert dataset_name_for(data) == f"helix-csm-eval-v{data['version']}"
    changed = json.loads(json.dumps(data))
    changed["queries"][0]["query"] += "?"
    assert test_set_hash(changed) != test_set_hash(data)


def _fake_result(qid, diff, scores, ret_ms=100.0, gen_ms=900.0):
    return {
        "run": SimpleNamespace(outputs={"answer": "a", "retrieval_latency_ms": ret_ms,
                                        "generation_latency_ms": gen_ms,
                                        "retrieved_doc_ids": []}, error=None),
        "example": SimpleNamespace(inputs={"question": "q", "qid": qid},
                                   outputs={"expected_sources": []},
                                   metadata={"qid": qid, "difficulty": diff, "category": "api"}),
        "evaluation_results": {"results": [SimpleNamespace(key=k, score=v, comment=None)
                                           for k, v in scores.items()]},
    }


def test_run_evals_aggregation_and_gates():
    from ops.langsmith.run_evals import aggregate, gates, row_from_result
    rows = [row_from_result(_fake_result("Q1", "easy", {"faithfulness": 1.0, "csm_resolution_quality": 1.0})),
            row_from_result(_fake_result("Q2", "hard", {"faithfulness": 0.9, "csm_resolution_quality": 0.6}))]
    s = aggregate(rows)
    assert s["overall"]["faithfulness"] == pytest.approx(0.95)
    assert s["by_difficulty"]["hard"]["n"] == 1
    g = gates(s, baseline={"overall": {"faithfulness": 0.88}})
    assert g["custom_judge_gte_0.75"] is True
    assert g["faithfulness_delta"] == pytest.approx(0.07) and g["faithfulness_delta_gte_0.05"]


def test_calibration_stats():
    from ops.langsmith.calibrate_judge import calibration_stats, cohen_kappa
    perfect = calibration_stats([1, 0.5, 0, 1], [1, 0.5, 0, 1])
    assert perfect["mae"] == 0 and perfect["cohen_kappa"] == 1.0
    assert cohen_kappa([True, False], [False, True]) < 0


# ---------------------------------------------------------------------------------------
# Phoenix span attributes (in-memory OpenTelemetry exporter)
# ---------------------------------------------------------------------------------------

def test_retriever_span_carries_doc_ids_scores_and_chunk_sizes():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from ops.phoenix.instrument import set_retrieval_attributes

    exporter = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    chunks = [dict(chunk("product-docs/a.md", 0, "x" * 50), score=0.9),
              dict(chunk("product-docs/b.md", 0, "y" * 500), score=0.4)]
    with tp.get_tracer("t").start_as_current_span("retrieve") as span:
        set_retrieval_attributes(span, chunks)
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs["openinference.span.kind"] == "RETRIEVER"
    assert attrs["retrieval.documents.0.document.id"] == "product-docs/a.md::0"
    assert attrs["retrieval.documents.1.document.score"] == 0.4
    assert attrs["rag.retrieval.short_chunks"] == 1
    assert tuple(attrs["rag.retrieval.chunk_lengths"]) == (50, 500)


def test_phoenix_local_stats():
    from ops.phoenix.instrument import local_stats
    rows = [{"chunk_lengths": [50, 600], "source_hit": True, "abstained": False,
             "retrieval_ms": 300.0, "generation_ms": 1500.0},
            {"chunk_lengths": [150, 700], "source_hit": False, "abstained": True,
             "retrieval_ms": 500.0, "generation_ms": 1700.0}]
    s = local_stats(rows)
    assert s["source_hit_rate"] == 0.5 and s["abstention_rate"] == 0.5
    assert s["retrieved_chunks_under_200_pct"] == 0.5


def test_crashed_evaluators_are_counted_not_silently_dropped():
    from ops.langsmith.run_evals import aggregate, gates, row_from_result
    ok = _fake_result("Q1", "easy", {"faithfulness": 1.0, "csm_resolution_quality": 1.0})
    bad = _fake_result("Q2", "easy", {"csm_resolution_quality": 0.8})
    bad["evaluation_results"]["results"].append(
        SimpleNamespace(key="faithfulness", score=None, comment="RateLimitError", extra={"error": True}))
    rows = [row_from_result(ok), row_from_result(bad)]
    assert rows[1]["evaluator_errors"] == ["faithfulness"]
    s = aggregate(rows)
    assert s["scored_n"]["faithfulness"] == 1 and s["n"] == 2
    assert gates(s, None)["all_examples_scored"] is False


def test_scores_are_rounded_for_langsmith():
    v = FaithfulnessVerdict(claims=[ClaimCheck(claim=c, supported=s, reason="r")
                                    for c, s in [("a", True), ("b", True), ("c", False)]])
    assert score_faithfulness("x", ["ctx"], judge=fake_judge(v))["score"] == 0.6667
    ref = {"expected_sources": ["a", "b", "c"]}
    assert source_recall({}, {"retrieved_doc_ids": ["a"]}, ref)["score"] == 0.3333


def test_ingest_fails_fast_on_wrong_corpus_path(tmp_path):
    from ops.ingest import load_and_chunk
    with pytest.raises(SystemExit, match="Corpus not found"):
        load_and_chunk(str(tmp_path / "nope"))


# ---------------------------------------------------------------------------------------
# v3: prompt variants + comparison tool
# ---------------------------------------------------------------------------------------

def test_module1_configs_keep_module1_prompt():
    for name in ("v1", "v2", "v2-vector"):
        assert get_config(name).prompt == "module1"
    assert get_config("v3").prompt == "grounded"
    assert get_config("v3").namespace == get_config("v2").namespace  # no re-ingestion needed


def test_grounded_prompt_extends_module1_prompt_without_dropping_rules():
    from src.generate import SYSTEM_PROMPT
    from ops.generation import GROUNDED_SYSTEM_PROMPT
    for rule in ("1. Base your answer strictly", "2. If the context does not contain",
                 "3. In `sources`", "4. Set `confidence`"):
        assert rule in GROUNDED_SYSTEM_PROMPT
    assert "5. Every step" in GROUNDED_SYSTEM_PROMPT and "{context}" in GROUNDED_SYSTEM_PROMPT
    assert GROUNDED_SYSTEM_PROMPT.index("7.") < GROUNDED_SYSTEM_PROMPT.index("{context}")
    assert len(GROUNDED_SYSTEM_PROMPT) > len(SYSTEM_PROMPT)


def test_module1_variant_routes_to_unchanged_src_generate(monkeypatch):
    import ops.generation as G
    calls = []
    monkeypatch.setattr(G, "module1_generate", lambda q, c: calls.append(q) or "ok")
    assert G.generate_for("module1", "q?", []) == "ok" and calls == ["q?"]
    with pytest.raises(ValueError):
        G.generate_for("nope", "q", [])


def test_compare_runs_shows_zero_scores():
    from ops.langsmith.compare_runs import diff_rows
    a = {"Q1": {"scores": {"faithfulness": 0.0}, "comments": {}},
         "Q2": {"scores": {"faithfulness": 1.0}, "comments": {}}}
    b = {"Q1": {"scores": {"faithfulness": 1.0}, "comments": {"faithfulness": "3/3"}},
         "Q2": {"scores": {"faithfulness": 1.0}, "comments": {}}}
    rows = diff_rows(a, b, "faithfulness")
    assert [r[0] for r in rows] == ["Q1"] and rows[0][1] == 0.0


# ---------------------------------------------------------------------------------------
# faithfulness: abstentions are never counted as unsupported claims
# ---------------------------------------------------------------------------------------

def test_abstention_statement_extracted_as_claim_scores_one():
    """Regression: the judge scored Q017/Q033/Q044's abstention as 0/1 claims supported."""
    v = FaithfulnessVerdict(claims=[ClaimCheck(
        claim="The provided context doesn't cover this.", supported=False, reason="not in context")])
    out = score_faithfulness("The provided context doesn't cover this.", ["ctx"], judge=fake_judge(v))
    assert out["score"] == 1.0 and "ignored" in out["comment"]


def test_partial_answer_ignores_only_the_abstention_part():
    v = FaithfulnessVerdict(claims=[
        ClaimCheck(claim="Pro includes SSO.", supported=True, reason="ctx"),
        ClaimCheck(claim="Pro costs $500.", supported=False, reason="no"),
        ClaimCheck(claim="The context does not mention the Enterprise price.", supported=False, reason="n/a")])
    out = score_faithfulness("...", ["ctx"], judge=fake_judge(v))
    assert out["score"] == 0.5


def test_real_unsupported_claims_still_count():
    v = FaithfulnessVerdict(claims=[ClaimCheck(claim="Go to account settings.", supported=False, reason="no")])
    assert score_faithfulness("...", ["ctx"], judge=fake_judge(v))["score"] == 0.0
