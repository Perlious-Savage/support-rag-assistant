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
    """Lowercased content words (naive plural stripping) used for extractive overlap scoring."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in STOPWORDS}


def extractive_answer(question: str, chunks: list[dict]) -> dict:
    """Offline fallback: return the retrieved sentence with the most word overlap with the question.

    A sentence also "contains" its doc name and section headings, so "what is the auth policy?"
    matches sentences in auth_policy.md even though "auth" only appears in the filename.
    """
    q = _words(question)
    best, best_overlap = None, 0
    for c in chunks:
        if c["score"] < MIN_SCORE:
            continue
        headings = " ".join(l for l in c["text"].splitlines() if l.lstrip().startswith("#"))
        context = _words(Path(c["doc_id"]).stem.replace("_", " ") + " " + headings)
        for sent in re.split(r"(?<=[.!?])\s+|\n", c["text"]):
            sent = sent.strip().lstrip("-* ").strip()
            if not sent or sent.startswith("#"):
                continue
            overlap = len(q & (_words(sent) | context))
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
        return {**refusal("low_retrieval_score"), "mode": "threshold"}
    if extractive or not llm.available():
        return {**extractive_answer(question, chunks), "mode": "extractive"}
    try:
        data = llm.complete_json(build_prompt(question, chunks), stage="generate", item_id=question_id,
                                 input_artifacts=["retrieval_results.json", "prompts/answer.txt"],
                                 output_artifact="answers.json")
    except (RuntimeError, ValueError) as e:  # every LLM tier failed (e.g. quota) or unparseable twice
        print(f"WARNING: LLM unavailable, using extractive fallback: {str(e)[:200]}")
        return {**extractive_answer(question, chunks), "mode": "extractive_fallback"}
    return {**_normalize_llm_output(data), "mode": "llm"}
