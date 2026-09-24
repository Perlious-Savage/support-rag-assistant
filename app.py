"""CLI: ask one question against docs/.

Usage: python app.py --question "How long does a SEPA withdrawal take?" [--extractive]
"""
import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

from generation import generate_answer
from obs import log_event
from retrieval import build_index, chunk_documents, load_documents, retrieve
from support_check import enforce

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=4)
def load_corpus(docs_dir: Path) -> tuple:
    """Load, chunk and index docs once per folder (the server reuses it; restart to pick up doc edits)."""
    chunks = chunk_documents(load_documents(docs_dir))
    return chunks, build_index(chunks)


def ask(question: str, docs_dir: Path = ROOT / "docs", top_k: int = 4, extractive: bool = False) -> dict:
    """Retrieve -> generate -> validate for one question; returns answer, citations, supported, sources."""
    chunks, index = load_corpus(Path(docs_dir).resolve())
    hits = retrieve(question, chunks, index, top_k)
    raw = generate_answer(question, hits, question_id="cli", extractive=extractive)
    answer, report = enforce(raw, hits)
    log_event("cli_question_answered", supported=answer["supported"], errors=report["errors"])
    return {
        "question": question,
        "answer": answer["answer"],
        "citations": answer["citations"],
        "supported": answer["supported"],
        "mode": raw.get("mode"),  # llm | extractive | extractive_fallback | threshold
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
