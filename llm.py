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
# Backend chosen at runtime in the web UI. None = use .env tiers; [] = no LLM (extractive only).
# ponytail: one process-wide choice (fine for a local single-user tool); per-session config if it's ever shared
_override: list[dict] | None = None


def make_target(provider: str, base_url: str | None, model: str, api_key: str | None = None) -> dict:
    """Build an endpoint from UI input. provider='local' needs no key (Ollama/LM Studio ignore it)."""
    local = provider == "local"
    return {"provider": provider, "base_url": base_url or None, "api_key": api_key or "local", "model": model,
            "tier": "user", "fallback": False, "timeout": 180 if local else 60}


def set_override(targets: list[dict] | None) -> None:
    """Use exactly these endpoints instead of .env ([] = no LLM); None restores the .env config."""
    global _override
    _override = targets


def ping(target: dict) -> str | None:
    """Send a tiny prompt to one endpoint; return None if it answers, else a short error message."""
    from openai import OpenAI
    try:
        client = OpenAI(api_key=target["api_key"], base_url=target["base_url"], timeout=target["timeout"], max_retries=0)
        client.chat.completions.create(model=target["model"], max_tokens=5,
                                       messages=[{"role": "user", "content": "Reply with: OK"}])
        return None
    except Exception as e:  # never echo the key back, even if the provider's error message contains it
        return f"{type(e).__name__}: {str(e).replace(target['api_key'], '***')[:300]}"


def status() -> dict:
    """Describe the active and .env-configured endpoints (never includes keys)."""
    describe = lambda ts: [{k: t[k] for k in ("tier", "provider", "model", "base_url")} for t in ts]
    return {"source": "env" if _override is None else "ui", "active": describe(_targets()),
            "env": describe(_env_targets())}


def _targets() -> list[dict]:
    """Endpoints to try in order: the UI override if one is set, else the .env tiers."""
    return _env_targets() if _override is None else _override


def _env_targets() -> list[dict]:
    """Return .env-configured endpoints in priority order (primary, fallback, local).

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

    use_cache = _override is None  # cache is keyed by prompt only, so it is valid only for the .env models
    if use_cache and cache_file.exists():
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
                code = getattr(e, "status_code", None)
                daily_quota_gone = code == 429 and "PerDay" in str(e)  # retrying won't help today
                # no HTTP status = network error: worth retrying for a cloud API, not for a local server that's down
                transient = code in RETRYABLE or (code is None and target["provider"] != "local")
                if transient and not daily_quota_gone and attempt <= len(BACKOFF_SECONDS):
                    time.sleep(BACKOFF_SECONDS[attempt - 1])
                    continue
                break
            if use_cache:
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
