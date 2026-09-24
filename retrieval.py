"""Retrieval: load documents, chunk them with metadata, embed into an index, return top-k chunks."""
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

DOC_EXTENSIONS = {".md", ".txt"}
MAX_CHUNK_CHARS = 600
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"


def load_documents(docs_dir: Path) -> list[dict]:
    """Read every .md/.txt file in docs_dir -> [{doc_id, text}]; doc_id is the filename."""
    docs_dir = Path(docs_dir)
    files = sorted(p for p in docs_dir.glob("*") if p.suffix.lower() in DOC_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No .md/.txt documents found in {docs_dir.resolve()}")
    return [{"doc_id": p.name, "text": p.read_text(encoding="utf-8")} for p in files]


def _split_long(section: str, max_chars: int) -> list[str]:
    """Split an oversized section on blank lines, greedily merging paragraphs up to max_chars."""
    if len(section) <= max_chars:
        return [section]
    out, buf = [], ""
    for para in (p.strip() for p in re.split(r"\n\s*\n", section)):
        if not para:
            continue
        if buf and len(buf) + len(para) + 2 > max_chars:
            out.append(buf)
            buf = ""
        buf = f"{buf}\n\n{para}" if buf else para
    # ponytail: a single paragraph longer than max_chars stays whole; add sentence splitting if docs get long
    return out + ([buf] if buf else [])


def chunk_documents(docs: list[dict], max_chars: int = MAX_CHUNK_CHARS) -> list[dict]:
    """Split docs on markdown headings (then paragraphs) -> [{doc_id, chunk_id, text}].

    Heading-only sections (e.g. a title right before a subheading) are merged into the next section.
    """
    chunks = []
    for doc in docs:
        stem = Path(doc["doc_id"]).stem
        pieces, carry = [], ""
        for section in re.split(r"(?m)^(?=#{1,6}\s)", doc["text"]):
            section = (carry + section).strip()
            carry = ""
            if not section:
                continue
            if all(line.lstrip().startswith("#") for line in section.splitlines() if line.strip()):
                carry = section + "\n"
                continue
            pieces.extend(_split_long(section, max_chars))
        if carry.strip():
            pieces.append(carry.strip())
        chunks += [{"doc_id": doc["doc_id"], "chunk_id": f"{stem}_{i}", "text": t} for i, t in enumerate(pieces)]
    return chunks


@lru_cache(maxsize=1)
def _embedder():
    """Load the local fastembed model once (downloads on first use, then cached on disk)."""
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=EMBEDDING_MODEL)


def _normalize(m: np.ndarray) -> np.ndarray:
    """L2-normalize rows so dot product = cosine similarity."""
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def build_index(chunks: list[dict]) -> np.ndarray:
    """Embed chunk texts -> normalized matrix (row i = chunks[i])."""
    if not chunks:
        raise ValueError("Cannot build an index from zero chunks")
    return _normalize(np.array(list(_embedder().passage_embed([c["text"] for c in chunks]))))


def retrieve(question: str, chunks: list[dict], index: np.ndarray, top_k: int = 4) -> list[dict]:
    """Return the top_k chunks by cosine similarity: [{doc_id, chunk_id, score, text}], best first."""
    q = _normalize(np.array(list(_embedder().query_embed([question]))))[0]
    scores = index @ q
    order = np.argsort(-scores)[:top_k]
    return [{**chunks[i], "score": round(float(scores[i]), 4)} for i in order]
