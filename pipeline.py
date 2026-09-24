"""Batch pipeline: load_documents -> chunk -> index -> retrieve -> generate -> validate.

Writes retrieval_results.json, answers.json, validation_report.json and run_manifest.json.
Usage: python pipeline.py [--extractive] [--docs docs] [--questions questions.json] [--top-k 4]
"""
import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import llm
from generation import MIN_SCORE, PROMPT_TEMPLATE, generate_answer
from obs import PIPELINE_LOG, log_event, utc_now
from retrieval import EMBEDDING_MODEL, build_index, chunk_documents, load_documents, retrieve
from support_check import enforce

ROOT = Path(__file__).resolve().parent
STAGES = ["load_documents", "chunk", "index", "retrieve", "generate", "validate"]


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    """Write pretty UTF-8 JSON."""
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_questions(path: Path) -> list[dict]:
    """Load and sanity-check questions.json: a list of {id, question, ...}."""
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not all(isinstance(q, dict) and q.get("id") and q.get("question")
                                                  for q in questions):
        raise ValueError(f"{path} must be a JSON list of objects with 'id' and 'question'")
    return questions


def run(docs_dir: Path, questions_path: Path, out_dir: Path, top_k: int, extractive: bool) -> dict:
    """Run all stages in order and write every artifact; returns the validation summary."""
    for f in (PIPELINE_LOG, llm.CALL_LOG):  # fresh logs per run so they describe this run's artifacts
        f.unlink(missing_ok=True)
    manifest = {"started_at": utc_now(), "stages": [], "inputs": {},
                "mode": "extractive" if extractive or not llm.available() else "llm",
                "config": {"top_k": top_k, "min_score": MIN_SCORE}}

    @contextmanager
    def stage(name: str):
        """Print, time and record one stage; stages must run in STAGES order."""
        expected = STAGES[len(manifest["stages"])]
        assert name == expected, f"stage order violated: got {name}, expected {expected}"
        print(f"== STAGE: {name}", flush=True)
        entry = {"stage": name, "started_at": utc_now()}
        yield
        entry["ended_at"] = utc_now()
        manifest["stages"].append(entry)

    with stage("load_documents"):
        docs = load_documents(docs_dir)
        questions = load_questions(questions_path)
        for p in [*sorted(Path(docs_dir).glob("*")), Path(questions_path), PROMPT_TEMPLATE]:
            if p.is_file():
                manifest["inputs"][p.resolve().relative_to(ROOT).as_posix() if p.resolve().is_relative_to(ROOT)
                                   else str(p)] = sha256_file(p)
        log_event("documents_loaded", count=len(docs), doc_ids=[d["doc_id"] for d in docs])

    with stage("chunk"):
        chunks = chunk_documents(docs)
        log_event("chunks_created", count=len(chunks))

    with stage("index"):
        index = build_index(chunks)
        log_event("index_built", model=EMBEDDING_MODEL, shape=list(index.shape))

    with stage("retrieve"):
        retrieval_results = []
        for q in questions:
            hits = retrieve(q["question"], chunks, index, top_k)
            retrieval_results.append({"question_id": q["id"], "question": q["question"], "retrieved_chunks": hits})
            log_event("retrieval_executed", question_id=q["id"], top_doc=hits[0]["doc_id"], top_score=hits[0]["score"])
        write_json(out_dir / "retrieval_results.json", retrieval_results)

    with stage("generate"):
        raw_answers = []
        for q, r in zip(questions, retrieval_results):
            ans = generate_answer(q["question"], r["retrieved_chunks"], question_id=q["id"], extractive=extractive)
            raw_answers.append(ans)
            log_event("answer_generated", question_id=q["id"], supported=ans["supported"], reason=ans["reason"])

    with stage("validate"):
        answers, results = [], []
        for q, r, ans in zip(questions, retrieval_results, raw_answers):
            final, report = enforce(ans, r["retrieved_chunks"])
            expected = q.get("expected_behavior")
            results.append({"question_id": q["id"], **report, "expected_behavior": expected,
                            "final_supported": final["supported"],
                            "behavior_match": None if expected is None else (expected == "answerable") == final["supported"]})
            answers.append({"question_id": q["id"], "question": q["question"], **final, "mode": ans.get("mode"),
                            "retrieved_doc_ids": sorted({c["doc_id"] for c in r["retrieved_chunks"]}),
                            "retrieved_chunk_ids": [c["chunk_id"] for c in r["retrieved_chunks"]]})
            log_event("validation_" + ("passed" if report["passed"] else "failed"), question_id=q["id"],
                      errors=report["errors"], downgraded=report["downgraded"])
        matches = [x["behavior_match"] for x in results if x["behavior_match"] is not None]
        summary = {"total": len(results), "passed": sum(x["passed"] for x in results),
                   "failed": sum(not x["passed"] for x in results),
                   "downgraded": sum(x["downgraded"] for x in results),
                   "supported": sum(a["supported"] for a in answers),
                   "refused": sum(not a["supported"] for a in answers),
                   "expected_behavior_match": f"{sum(matches)}/{len(matches)}"}
        write_json(out_dir / "answers.json", answers)
        write_json(out_dir / "validation_report.json", {"summary": summary, "results": results})

    manifest.update(ended_at=utc_now(), models={"embedding": EMBEDDING_MODEL, **llm.configured_models(),
                                                "llm_models_used": sorted(llm.MODELS_USED)})
    write_json(out_dir / "run_manifest.json", manifest)
    return summary


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", default=ROOT / "docs", type=Path)
    ap.add_argument("--questions", default=ROOT / "questions.json", type=Path)
    ap.add_argument("--out", default=ROOT, type=Path)
    ap.add_argument("--top-k", default=4, type=int)
    ap.add_argument("--extractive", action="store_true", help="skip the LLM; answer extractively (no API key needed)")
    args = ap.parse_args()
    try:
        summary = run(args.docs, args.questions, args.out, args.top_k, args.extractive)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
