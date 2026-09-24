"""Generation: build the grounded prompt, call the LLM (or extractive fallback), normalize the answer."""
import os
import re
from pathlib import Path

import llm

ROOT = Path(__file__).resolve().parent
PROMPT_TEMPLATE = ROOT / "prompts" / "answer.txt"
REFUSAL = "I can't answer that from the available documentation."
# Stretch: refuse weak retrievals before spending an LLM call. Calibrated on the sample corpus:
# answerable questions score >= 0.76, off-topic ones ~0.53 (bge-small cosine).
MIN_SCORE = float(os.getenv("MIN_SCORE") or 0.6)
STOPWORDS = set("a an and are be can do does for from how in is it of on or the to what when which who why with".split())


def refusal(reason: str) -> dict:
    """Standard unsupported answer: fixed refusal text, no citations."""
    return {"answer": REFUSAL, "citations": [], "supported": False, "evidence": [], "reason": reason}


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Fill the prompt template with the question and numbered, doc-tagged context passages."""
    context = "\n\n".join(f"[{i}] (doc_id: {c['doc_id']})\n{c['text']}" for i, c in enumerate(chunks, 1))
    return PROMPT_TEMPLATE.read_text(encoding="utf-8").replace("{context}", context).replace("{question}", question)


def _normalize_llm_output(data: dict) -> dict:
    """Coerce model JSON into the answer schema; anything not explicitly supported becomes a refusal."""
    as_list = lambda v: [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []
    answer = str(data.get("answer") or "").strip()
    if data.get("supported") is not True:
        return refusal("model_marked_unsupported")
    return {"answer": answer, "citations": sorted(set(as_list(data.get("citations")))),
            "supported": True, "evidence": as_list(data.get("evidence")), "reason": None}


def _words(text: str) -> set[str]:
    """Lowercased content words used for extractive overlap scoring."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def extractive_answer(question: str, chunks: list[dict]) -> dict:
    """Offline fallback: return the retrieved sentence with the most word overlap with the question."""
    q = _words(question)
    best, best_overlap = None, 0
    for c in chunks:
        if c["score"] < MIN_SCORE:
            continue
        for sent in re.split(r"(?<=[.!?])\s+|\n", c["text"]):
            sent = sent.strip().lstrip("-* ").strip()
            if not sent or sent.startswith("#"):
                continue
            overlap = len(q & _words(sent))
            if overlap > best_overlap:  # strict > keeps the higher-ranked chunk on ties
                best, best_overlap = (sent, c["doc_id"]), overlap
    # ponytail: word overlap can't tell "answers the question" from "mentions the topic"; LLM mode handles that
    if best is None or best_overlap < 2:
        return refusal("no_extractive_match")
    return {"answer": best[0], "citations": [best[1]], "supported": True, "evidence": [best[0]], "reason": None}


def generate_answer(question: str, chunks: list[dict], *, question_id: str | None = None,
                    extractive: bool = False) -> dict:
    """Answer from retrieved chunks only. Refuses weak retrievals; uses extractive mode if no LLM is configured."""
    if not chunks or chunks[0]["score"] < MIN_SCORE:
        return refusal("low_retrieval_score")
    if extractive or not llm.available():
        return extractive_answer(question, chunks)
    data = llm.complete_json(build_prompt(question, chunks), stage="generate", item_id=question_id,
                             input_artifacts=["retrieval_results.json", "prompts/answer.txt"],
                             output_artifact="answers.json")
    return _normalize_llm_output(data)
