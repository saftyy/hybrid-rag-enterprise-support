"""
expectations.py — The Great Expectations "data contract" for the chunk table that is
about to be embedded and written to Pinecone.

One row = one chunk. Columns validated:
    id, text, doc_id, source, category, doc_type, chunk_index   (from ingestion)
    text_len                                                    (derived)
    embedding_dim, embedding_has_nan                            (only when embeddings exist)

Each expectation below states the production failure it guards against. Thresholds are
constants so the README / monitoring spec can quote them and tests can import them.
"""

from __future__ import annotations

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

from ops.data_quality.transforms import PII_PATTERNS

# --- Thresholds (the contract) ------------------------------------------------------------
EXPECTED_EMBEDDING_DIM = 1536          # text-embedding-3-small; Pinecone index dimension
MIN_CHUNK_CHARS = 200                  # below this a chunk is usually a bare heading
MAX_CHUNK_CHARS = 3000                 # tickets are kept whole up to 3000 chars
CHUNK_LEN_MOSTLY = 0.95                # tolerate 5% outliers (e.g. a genuinely short doc)
MEDIAN_CHUNK_RANGE = (300, 1500)       # distribution check, not just per-row bounds
HARD_MIN_CHARS, HARD_MAX_CHARS = 20, 4000   # no exceptions: empty/garbage or runaway chunks
MIN_DISTINCT_DOCS = 90                 # corpus has 95 readable docs; fewer = silent drop
VALID_CATEGORIES = ["product-docs", "runbooks", "tickets"]
VALID_DOC_TYPES = ["markdown", "pdf", "html"]
DOC_ID_REGEX = r"^(product-docs|runbooks|tickets)/.+\.(md|pdf|html)$"
REQUIRED_METADATA = ["id", "text", "doc_id", "source", "category", "doc_type", "chunk_index"]

SUITE_NAME = "helix_chunks_contract"


def build_expectations(include_embeddings: bool) -> list[tuple[str, gxe.Expectation]]:
    """Returns (group, expectation) pairs. `group` is used to organise the report."""
    ex: list[tuple[str, gxe.Expectation]] = []

    # 1. Chunk length distribution --------------------------------------------------------
    # Guards against: header-only fragments (poor embeddings, "right doc, wrong chunk"),
    # and oversized chunks that blow the context window / dilute similarity.
    ex.append(("chunk_length", gxe.ExpectColumnValueLengthsToBeBetween(
        column="text", min_value=MIN_CHUNK_CHARS, max_value=MAX_CHUNK_CHARS,
        mostly=CHUNK_LEN_MOSTLY)))
    ex.append(("chunk_length", gxe.ExpectColumnMedianToBeBetween(
        column="text_len", min_value=MEDIAN_CHUNK_RANGE[0], max_value=MEDIAN_CHUNK_RANGE[1])))
    ex.append(("chunk_length", gxe.ExpectColumnValueLengthsToBeBetween(
        column="text", min_value=HARD_MIN_CHARS, max_value=HARD_MAX_CHARS)))

    # 2. Metadata completeness ------------------------------------------------------------
    # Guards against: answers citing sources that can't be resolved, and filters by
    # category silently dropping chunks with null metadata.
    for col in REQUIRED_METADATA:
        ex.append(("metadata", gxe.ExpectColumnValuesToNotBeNull(column=col)))
    ex.append(("metadata", gxe.ExpectColumnValuesToBeUnique(column="id")))
    ex.append(("metadata", gxe.ExpectColumnValuesToBeInSet(
        column="category", value_set=VALID_CATEGORIES)))
    ex.append(("metadata", gxe.ExpectColumnValuesToBeInSet(
        column="doc_type", value_set=VALID_DOC_TYPES)))
    ex.append(("metadata", gxe.ExpectColumnValuesToMatchRegex(
        column="doc_id", regex=DOC_ID_REGEX)))
    # Guards against: a loader breaking and a whole folder quietly disappearing.
    ex.append(("metadata", gxe.ExpectColumnUniqueValueCountToBeBetween(
        column="doc_id", min_value=MIN_DISTINCT_DOCS)))

    # 3. Embedding dimension consistency --------------------------------------------------
    # Guards against: someone changing EMBEDDING_MODEL (e.g. to -3-large, 3072 dims) without
    # rebuilding the index — Pinecone rejects the upsert, or worse, a mixed index.
    if include_embeddings:
        ex.append(("embeddings", gxe.ExpectColumnValuesToBeInSet(
            column="embedding_dim", value_set=[EXPECTED_EMBEDDING_DIM])))
        ex.append(("embeddings", gxe.ExpectColumnValuesToBeInSet(
            column="embedding_has_nan", value_set=[False])))

    # 4. No PII ---------------------------------------------------------------------------
    # Guards against: customer emails/phones being embedded, stored in Pinecone metadata,
    # and quoted back to a CSM (or anyone with index access).
    ex.append(("pii", gxe.ExpectColumnValuesToNotMatchRegexList(
        column="text", regex_list=list(PII_PATTERNS.values()))))

    return ex


def chunks_to_dataframe(chunks: list[dict], embeddings: list[list[float]] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(chunks)
    for col in REQUIRED_METADATA:
        if col not in df.columns:
            df[col] = None
    df["text_len"] = df["text"].fillna("").str.len()
    if embeddings is not None:
        df["embedding_dim"] = [len(e) if e is not None else 0 for e in embeddings]
        df["embedding_has_nan"] = [
            any(v != v for v in e) if e is not None else True for e in embeddings
        ]
    return df


def run_suite(df: pd.DataFrame, include_embeddings: bool) -> list[dict]:
    """Validate `df` and return one plain-dict result per expectation."""
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas("helix_ingestion")
    asset = source.add_dataframe_asset(name="chunks")
    batch_def = asset.add_batch_definition_whole_dataframe("all_chunks")

    pairs = build_expectations(include_embeddings)
    suite = context.suites.add(gx.ExpectationSuite(name=SUITE_NAME))
    group_by_id: dict[str, str] = {}
    for group, e in pairs:
        added = suite.add_expectation(e)
        group_by_id[str(added.id)] = group

    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    validation = batch.validate(suite, result_format="SUMMARY")

    results = []
    for r in validation.results:
        cfg = r.expectation_config
        res = r.result or {}
        results.append({
            "group": group_by_id.get(str(cfg.id), "other"),
            "expectation": cfg.type,
            "column": cfg.kwargs.get("column"),
            "kwargs": {k: v for k, v in cfg.kwargs.items()
                       if k not in ("column", "batch_id", "regex_list")},
            "success": bool(r.success),
            "observed_value": res.get("observed_value"),
            "unexpected_count": res.get("unexpected_count"),
            "unexpected_percent": res.get("unexpected_percent"),
            "element_count": res.get("element_count"),
            "exception": (r.exception_info or {}).get("exception_message")
            if isinstance(r.exception_info, dict) and r.exception_info.get("raised_exception")
            else None,
        })
    return results
