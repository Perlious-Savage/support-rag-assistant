"""Validation: deterministic citation/support rules applied to every answer (no LLM)."""
import re

from generation import REFUSAL, refusal


def _norm(text: str) -> str:
    """Normalize for verbatim matching: lowercase, drop markdown `*_, collapse whitespace."""
    return " ".join(re.sub(r"[`*_]", "", text).lower().split()).strip(" .")


def check_answer(answer: dict, retrieved: list[dict]) -> dict[str, bool]:
    """Run every support rule; returns {rule_name: passed}."""
    supported = answer.get("supported") is True
    citations = answer.get("citations") or []
    retrieved_docs = {c["doc_id"] for c in retrieved}
    cited_text = _norm(" ".join(c["text"] for c in retrieved if c["doc_id"] in citations))
    evidence = answer.get("evidence") or []
    return {
        "answer_not_empty": bool(str(answer.get("answer") or "").strip()),
        "supported_has_citation": not supported or len(citations) >= 1,
        "citations_in_retrieved": all(c in retrieved_docs for c in citations),
        "evidence_in_cited_chunks": not supported or (bool(evidence) and all(_norm(e) in cited_text for e in evidence)),
        "unsupported_marked": supported or str(answer.get("answer", "")).startswith(REFUSAL),
    }


def enforce(answer: dict, retrieved: list[dict]) -> tuple[dict, dict]:
    """Check an answer and fail closed: a supported answer that breaks a rule becomes a refusal.

    Returns (final_answer, report_entry).
    """
    checks = check_answer(answer, retrieved)
    errors = [name for name, ok in checks.items() if not ok]
    final = answer
    if errors:
        final = refusal("failed_support_check")
    return final, {"passed": not errors, "checks": checks, "errors": errors, "downgraded": bool(errors)}
