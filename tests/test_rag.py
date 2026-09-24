"""Focused tests for chunking, retrieval, refusal and the support check (no network: extractive mode)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation import REFUSAL, generate_answer  # noqa: E402
from retrieval import build_index, chunk_documents, load_documents, retrieve  # noqa: E402
from support_check import enforce  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"


@pytest.fixture(scope="module")
def corpus():
    """Chunks + index over the sample docs (built once)."""
    chunks = chunk_documents(load_documents(DOCS))
    return chunks, build_index(chunks)


def test_chunks_preserve_metadata(corpus):
    chunks, _ = corpus
    doc_ids = {p.name for p in DOCS.glob("*.md")}
    assert {c["doc_id"] for c in chunks} == doc_ids  # every doc produced chunks
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)  # chunk ids unique
    for c in chunks:
        assert c["chunk_id"].startswith(Path(c["doc_id"]).stem + "_")
        assert c["text"].strip()


def test_heading_only_sections_are_merged():
    chunks = chunk_documents([{"doc_id": "x.md", "text": "# Title\n\n## Sub\nBody text.\n\n## Two\nMore."}])
    assert [c["text"] for c in chunks] == ["# Title\n## Sub\nBody text.", "## Two\nMore."]


@pytest.mark.parametrize("question, expected_doc", [
    ("How many password reset emails can I request per hour?", "auth_policy.md"),
    ("What status code is returned when the API rate limit is exceeded?", "api_rate_limits.md"),
    ("How long does a SEPA withdrawal take?", "withdrawals.md"),
])
def test_retrieval_returns_expected_doc(corpus, question, expected_doc):
    chunks, index = corpus
    assert retrieve(question, chunks, index, top_k=4)[0]["doc_id"] == expected_doc


def test_off_topic_question_is_refused(corpus):
    chunks, index = corpus
    hits = retrieve("What is the company's parental leave policy?", chunks, index)
    ans = generate_answer("What is the company's parental leave policy?", hits, extractive=True)
    assert ans["supported"] is False and ans["answer"] == REFUSAL and ans["citations"] == []


def test_support_check_downgrades_citation_outside_retrieved_set(corpus):
    chunks, index = corpus
    hits = retrieve("How long does a SEPA withdrawal take?", chunks, index)
    fabricated = {"answer": "SEPA takes 1 business day.", "citations": ["hr_policy.md"], "supported": True,
                  "evidence": ["SEPA transfer: 1 business day."]}
    final, report = enforce(fabricated, hits)
    assert "citations_in_retrieved" in report["errors"]
    assert final["supported"] is False and final["answer"] == REFUSAL


def test_support_check_rejects_uncited_and_unquoted_answers(corpus):
    chunks, index = corpus
    hits = retrieve("How long does a SEPA withdrawal take?", chunks, index)
    _, report = enforce({"answer": "Instantly.", "citations": [], "supported": True, "evidence": []}, hits)
    assert {"supported_has_citation", "evidence_in_cited_chunks"} <= set(report["errors"])
    _, report = enforce({"answer": "Instantly.", "citations": ["withdrawals.md"], "supported": True,
                         "evidence": ["SEPA withdrawals are instant."]}, hits)
    assert report["errors"] == ["evidence_in_cited_chunks"]  # invented quote not in the text


def test_valid_grounded_answer_passes(corpus):
    chunks, index = corpus
    hits = retrieve("How long does a SEPA withdrawal take?", chunks, index)
    _, report = enforce({"answer": "1 business day.", "citations": ["withdrawals.md"], "supported": True,
                         "evidence": ["SEPA transfer: 1 business day."]}, hits)
    assert report["passed"]
