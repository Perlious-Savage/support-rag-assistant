"""Observability: append-only structured JSON-lines logs."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE_LOG = ROOT / "logs" / "pipeline.jsonl"


def utc_now() -> str:
    """Return the current time as an ISO-8601 UTC string."""
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record as a line to `path`, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_event(event: str, **fields) -> None:
    """Log one pipeline event (e.g. documents_loaded) to logs/pipeline.jsonl."""
    append_jsonl(PIPELINE_LOG, {"ts": utc_now(), "event": event, **fields})
