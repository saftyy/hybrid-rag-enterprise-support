"""Generation produces valid structured output."""

from src.generate import generate, RAGResponse


def test_generate_produces_valid_structured_output():
    context_chunks = [
        {
            "doc_id": "product-docs/test.md",
            "text": (
                "Workflows time out after 30 minutes by default. You can increase this "
                "in Settings > Workflows > Timeout."
            ),
        }
    ]
    result = generate("How do I change the workflow timeout?", context_chunks)

    assert isinstance(result, RAGResponse)
    assert isinstance(result.answer, str) and len(result.answer) > 0
    assert isinstance(result.sources, list)
    assert result.confidence in ("low", "medium", "high")


def test_generate_admits_uncertainty_with_no_context():
    result = generate("What is the CEO's favorite color?", [])

    assert isinstance(result, RAGResponse)
    assert result.confidence == "low"