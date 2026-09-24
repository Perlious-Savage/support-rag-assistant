"""CLI: ask one question against docs/.

Usage: python app.py --question "How long does a SEPA withdrawal take?" [--extractive]
"""
import argparse
import json
import sys
from pathlib import Path

from generation import generate_answer
from obs import log_event
from retrieval import build_index, chunk_documents, load_documents, retrieve
from support_check import enforce

ROOT = Path(__file__).resolve().parent


def ask(question: str, docs_dir: Path = ROOT / "docs", top_k: int = 4, extractive: bool = False) -> dict:
    """Retrieve -> generate -> validate for one question; returns answer, citations, supported, sources."""
    chunks = chunk_documents(load_documents(docs_dir))
    hits = retrieve(question, chunks, build_index(chunks), top_k)
    answer, report = enforce(generate_answer(question, hits, question_id="cli", extractive=extractive), hits)
    log_event("cli_question_answered", supported=answer["supported"], errors=report["errors"])
    return {
        "question": question,
        "answer": answer["answer"],
        "citations": answer["citations"],
        "supported": answer["supported"],
        "sources": [{"doc_id": h["doc_id"], "chunk_id": h["chunk_id"], "score": h["score"]} for h in hits],
        "validation": report,
    }


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--question", "-q", required=True)
    ap.add_argument("--docs", default=ROOT / "docs", type=Path)
    ap.add_argument("--top-k", default=4, type=int)
    ap.add_argument("--extractive", action="store_true", help="skip the LLM; answer extractively")
    args = ap.parse_args()
    if not args.question.strip():
        ap.error("--question must not be empty")
    try:
        print(json.dumps(ask(args.question, args.docs, args.top_k, args.extractive), indent=2, ensure_ascii=False))
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
