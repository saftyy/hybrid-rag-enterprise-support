"""
config.py — Named, versioned pipeline configurations.

Why this exists: the checkpoint requires (a) a reproducible Module 1 *baseline*
and (b) an improved system measured against it. Instead of editing src/ and
losing the baseline, every variant of the pipeline is a named config. Evals,
Phoenix runs and data-quality runs all take `--config <name>` and record it,
so any number in the report can be traced back to exactly what produced it.

    v1         Module 1 exactly as shipped: header-split chunks, hybrid (BM25 + vector) RRF,
               k=5, default Pinecone namespace. This is the baseline. src/ is untouched.
    v2         Fix for the weakness found in Phoenix + Great Expectations: tiny markdown
               chunks are merged (min 400 chars), customer PII is redacted at ingestion.
               Retrieval unchanged (hybrid) so the chunking fix is measured in isolation.
    v2-vector  v2 chunks + vector-only retrieval (Module 1's own compare_retrieval.py showed
               vector-only had better Hit@5 than hybrid on this corpus).

v2 / v2-vector live in a separate Pinecone *namespace* of the same index, so the
v1 baseline vectors are never overwritten and no extra index is needed.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    description: str
    chunks_path: Path
    namespace: str                                  # Pinecone namespace ("" = default)
    retrieval_mode: Literal["hybrid", "vector"]
    k: int = 5
    merge_min_chars: int = 0                        # 0 = no merging (Module 1 behaviour)
    merge_max_chars: int = 1500
    redact_pii: bool = False

    @property
    def index_name(self) -> str:
        return os.environ.get("PINECONE_INDEX_NAME", "helix-checkpoint-rag")

    def to_metadata(self) -> dict:
        d = asdict(self)
        d["chunks_path"] = str(self.chunks_path.relative_to(REPO_ROOT)).replace("\\", "/")
        d["index_name"] = self.index_name
        return d


CONFIGS: dict[str, PipelineConfig] = {
    "v1": PipelineConfig(
        name="v1",
        description="Module 1 baseline (header-split chunks, hybrid RRF, k=5)",
        chunks_path=REPO_ROOT / "evals" / "results" / "chunks.json",
        namespace="",
        retrieval_mode="hybrid",
    ),
    "v2": PipelineConfig(
        name="v2",
        description="Merged small markdown chunks (>=400 chars) + PII redaction, hybrid RRF, k=5",
        chunks_path=REPO_ROOT / "ops" / "data" / "chunks_v2.json",
        namespace="v2",
        retrieval_mode="hybrid",
        merge_min_chars=400,
        redact_pii=True,
    ),
    "v2-vector": PipelineConfig(
        name="v2-vector",
        description="v2 chunks + vector-only retrieval, k=5",
        chunks_path=REPO_ROOT / "ops" / "data" / "chunks_v2.json",
        namespace="v2",
        retrieval_mode="vector",
        merge_min_chars=400,
        redact_pii=True,
    ),
}


def get_config(name: str) -> PipelineConfig:
    try:
        return CONFIGS[name]
    except KeyError:
        raise SystemExit(f"Unknown config '{name}'. Choose from: {', '.join(CONFIGS)}")
