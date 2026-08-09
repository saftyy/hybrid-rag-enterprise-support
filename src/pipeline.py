"""
pipeline.py — End-to-end: query -> retrieve -> generate -> structured answer.
"""

from src.retrieve import retrieve
from src.generate import generate, RAGResponse


def answer_query(query: str, k: int = 5) -> RAGResponse:
    chunks = retrieve(query, k=k)
    return generate(query, chunks)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is Helix?"
    result = answer_query(q)
    print(result.model_dump_json(indent=2))
