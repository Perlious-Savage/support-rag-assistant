"""Replaceable LLM client: primary -> fallback model, retry with backoff, disk cache, call log.

Any OpenAI-compatible endpoint works (configured in .env). Swap this module to change provider.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from obs import append_jsonl, utc_now

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

CACHE_DIR = ROOT / "cache"
CALL_LOG = ROOT / "llm_calls.jsonl"
BACKOFF_SECONDS = [2, 4, 8, 16]
RETRYABLE = {429, 500, 502, 503, 504}
MODELS_USED: set[str] = set()


TIERS = ("LLM", "FALLBACK", "LOCAL")  # primary -> fallback -> local (e.g. Ollama)


def _targets() -> list[dict]:
    """Return configured endpoints in priority order (primary, fallback, local).

    LOCAL_* is any OpenAI-compatible local server (Ollama, LM Studio, llama.cpp) and needs no API key.
    """
    out = []
    for prefix in TIERS:
        base_url, model = os.getenv(f"{prefix}_BASE_URL") or None, os.getenv(f"{prefix}_MODEL")
        key = os.getenv(f"{prefix}_API_KEY") or ("local" if prefix == "LOCAL" else None)
        if key and model and (prefix != "LOCAL" or base_url):
            out.append({
                "provider": os.getenv(f"{prefix}_PROVIDER", "local" if prefix == "LOCAL" else "openai"),
                "base_url": base_url,
                "api_key": key,
                "model": model,
                "tier": prefix.lower(),
                "fallback": prefix != "LLM",
                "timeout": 180 if prefix == "LOCAL" else 60,  # CPU inference is slow
            })
    return out


def available() -> bool:
    """True if at least one LLM endpoint is configured."""
    return bool(_targets())


def configured_models() -> dict:
    """Model names (never keys) for the run manifest."""
    return {t["tier"]: f'{t["provider"]}/{t["model"]}' for t in _targets()}


def complete(prompt: str, *, stage: str, item_id: str | None, input_artifacts: list[str],
             output_artifact: str) -> str:
    """Send one prompt; return text. Cached by prompt hash; every call logged to llm_calls.jsonl."""
    from openai import OpenAI

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{prompt_hash}.json"
    log = {"stage": stage, "question_id": item_id, "prompt_hash": prompt_hash,
           "input_artifacts": input_artifacts, "output_artifact": output_artifact}

    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        MODELS_USED.add(cached["model"])
        append_jsonl(CALL_LOG, {**log, "timestamp": utc_now(), "provider": cached["provider"],
                                "model": cached["model"], "attempts": 0,
                                "fallback_used": cached["fallback"], "cached": True})
        return cached["text"]

    last_error = None
    for target in _targets():
        client = OpenAI(api_key=target["api_key"], base_url=target["base_url"], timeout=target["timeout"], max_retries=0)
        for attempt in range(1, len(BACKOFF_SECONDS) + 2):
            try:
                resp = client.chat.completions.create(
                    model=target["model"], temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.choices[0].message.content or ""
            except Exception as e:  # network / API errors: retry retryable ones, else next target
                last_error = e
                status = getattr(e, "status_code", None)
                daily_quota_gone = status == 429 and "PerDay" in str(e)  # retrying won't help today
                if (status in RETRYABLE or status is None) and not daily_quota_gone and attempt <= len(BACKOFF_SECONDS):
                    time.sleep(BACKOFF_SECONDS[attempt - 1])
                    continue
                break
            CACHE_DIR.mkdir(exist_ok=True)
            cache_file.write_text(json.dumps({"provider": target["provider"], "model": target["model"],
                                              "fallback": target["fallback"], "text": text}), encoding="utf-8")
            MODELS_USED.add(target["model"])
            append_jsonl(CALL_LOG, {**log, "timestamp": utc_now(), "provider": target["provider"],
                                    "model": target["model"], "attempts": attempt,
                                    "fallback_used": target["fallback"], "cached": False})
            return text
    raise RuntimeError(f"All LLM endpoints failed: {type(last_error).__name__}: {last_error}")


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, stripping ```json fences."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def complete_json(prompt: str, **meta) -> dict:
    """Ask for JSON; on parse failure retry once with the error appended to the prompt."""
    text = complete(prompt, **meta)
    try:
        return parse_json(text)
    except ValueError as e:  # json.JSONDecodeError is a ValueError
        retry = f"{prompt}\n\nYour previous reply was invalid JSON ({e}). Return ONLY the JSON object."
        return parse_json(complete(retry, **meta))
