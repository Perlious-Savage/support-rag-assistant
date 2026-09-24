"""Artifact validator: one PASS/FAIL line per requirement; exits non-zero on any FAIL.

Usage: python validate.py
"""
import json
import re
import sys
from pathlib import Path

from generation import REFUSAL
from pipeline import STAGES

ROOT = Path(__file__).resolve().parent
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record one check result."""
    results.append((name, bool(ok), detail))


def load(name: str):
    """Load a JSON artifact from the repo root, or None if missing/invalid."""
    try:
        return json.loads((ROOT / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def norm(text: str) -> str:
    """Same normalization as support_check: lowercase, strip markdown `*_, collapse whitespace."""
    return " ".join(re.sub(r"[`*_]", "", text).lower().split()).strip(" .")


def main() -> int:
    """Run all checks and print the report."""
    docs = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "docs").glob("*") if p.suffix in {".md", ".txt"}}
    questions, retrieval = load("questions.json"), load("retrieval_results.json")
    answers, report, manifest = load("answers.json"), load("validation_report.json"), load("run_manifest.json")

    # Required artifacts
    for f in ["questions.json", "retrieval_results.json", "answers.json", "validation_report.json",
              "run_manifest.json", "README.md"]:
        check(f"artifact exists: {f}", (ROOT / f).is_file() and (f.endswith(".md") or load(f) is not None))
    check("docs/ has >= 5 documents", len(docs) >= 5, f"{len(docs)} docs")
    if not all([questions, retrieval, answers, report, manifest]):
        return summarize()

    # questions.json
    beh = [q.get("expected_behavior") for q in questions]
    check("questions.json: >= 8 questions with id/question/expected_behavior",
          len(questions) >= 8 and all(q.get("id") and q.get("question") and q.get("expected_behavior") for q in questions))
    check("questions.json: >= 5 answerable, >= 3 unanswerable",
          beh.count("answerable") >= 5 and len(beh) - beh.count("answerable") >= 3)
    qids = [q["id"] for q in questions]

    # retrieval_results.json
    check("retrieval_results: one record per question, same ids", [r.get("question_id") for r in retrieval] == qids)
    chunk_fields_ok = all(r.get("retrieved_chunks") and all({"doc_id", "chunk_id", "score", "text"} <= set(c)
                          for c in r["retrieved_chunks"]) for r in retrieval)
    check("retrieval_results: chunks have doc_id, chunk_id, score, text", chunk_fields_ok)
    bad = [c["chunk_id"] for r in retrieval for c in r["retrieved_chunks"]
           if c["doc_id"] not in docs or norm(c["text"]) not in norm(docs[c["doc_id"]])]
    check("retrieval_results: chunk text exists verbatim in its source doc", not bad, ", ".join(bad[:5]))
    check("retrieval_results: scores sorted descending",
          all([c["score"] for c in r["retrieved_chunks"]] == sorted((c["score"] for c in r["retrieved_chunks"]), reverse=True)
              for r in retrieval))

    # answers.json
    retrieved_by_q = {r["question_id"]: r["retrieved_chunks"] for r in retrieval}
    check("answers: one record per question, same ids", [a.get("question_id") for a in answers] == qids)
    check("answers: fields question_id, answer, citations(list), supported(bool)",
          all(isinstance(a.get("answer"), str) and isinstance(a.get("citations"), list)
              and isinstance(a.get("supported"), bool) for a in answers))
    check("answers: answer text not empty", all(a["answer"].strip() for a in answers))
    bad = [a["question_id"] for a in answers if a["supported"] and not a["citations"]]
    check("answers: every supported answer has >= 1 citation", not bad, ", ".join(bad))
    bad = [a["question_id"] for a in answers
           if not set(a["citations"]) <= {c["doc_id"] for c in retrieved_by_q.get(a["question_id"], [])}]
    check("answers: cited docs appear in the retrieved set", not bad, ", ".join(bad))
    bad = [a["question_id"] for a in answers if not a["supported"] and not a["answer"].startswith(REFUSAL)]
    check("answers: unsupported answers clearly marked with refusal text", not bad, ", ".join(bad))
    bad = []
    for a in answers:
        if a["supported"]:
            cited = norm(" ".join(c["text"] for c in retrieved_by_q[a["question_id"]] if c["doc_id"] in a["citations"]))
            if not a.get("evidence") or any(norm(e) not in cited for e in a["evidence"]):
                bad.append(a["question_id"])
    check("answers: every evidence quote exists verbatim in a cited retrieved chunk", not bad, ", ".join(bad))

    # validation_report.json
    check("validation_report: summary + one result per question",
          "summary" in report and [r.get("question_id") for r in report.get("results", [])] == qids)
    check("validation_report: every result has passed/checks/errors",
          all({"passed", "checks", "errors"} <= set(r) for r in report.get("results", [])))

    # run_manifest.json: stage order + timestamps
    stages = manifest.get("stages", [])
    check("manifest: stages ran in order " + " -> ".join(STAGES), [s["stage"] for s in stages] == STAGES)
    times = [t for s in stages for t in (s["started_at"], s["ended_at"])]
    check("manifest: stage timestamps non-decreasing (retrieval written before generation)", times == sorted(times))
    check("manifest: sha256 recorded for every input doc + questions.json",
          all(f"docs/{d}" in manifest.get("inputs", {}) for d in docs) and "questions.json" in manifest.get("inputs", {}))

    # llm_calls.jsonl (only when the LLM was used)
    if manifest.get("mode") == "llm":
        lines = [json.loads(l) for l in (ROOT / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        fields = {"stage", "question_id", "timestamp", "provider", "model", "prompt_hash", "input_artifacts",
                  "output_artifact", "attempts", "fallback_used"}
        check("llm_calls.jsonl: every call has required fields", lines and all(fields <= set(l) for l in lines))
        n_refused_early = sum(a.get("reason") == "low_retrieval_score" for a in answers)
        check("llm_calls.jsonl: one call per question (except low-score refusals)",
              {l["question_id"] for l in lines} == {a["question_id"] for a in answers if a.get("reason") != "low_retrieval_score"},
              f"{len(lines)} calls, {n_refused_early} refused before LLM")
    return summarize()


def summarize() -> int:
    """Print PASS/FAIL lines; return exit code."""
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else ""))
    failed = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
