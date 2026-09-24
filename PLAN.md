# PLAN — Grounded Support-QA RAG Service

## Context
Per HANDOVER.md and the pasted question: build a small, runnable RAG service over local `docs/`. It retrieves passages, answers with citations, refuses when the docs don't support an answer, and deterministically validates every answer. The folder is empty except `.env` (never read or commit it), `.gitignore` and `HANDOVER.md`. The environment is already verified: `python` 3.13 with openai, python-dotenv and fastembed (bge-small cached). The LLM is Gemini through its OpenAI-compatible endpoint, with a primary and a fallback model.

## Pipeline stages (order enforced in code, timestamps in `run_manifest.json`)
`load_documents -> chunk -> index -> retrieve -> generate -> validate`

| Stage | Type | What it does | Output |
|---|---|---|---|
| 1 load_documents | DETERMINISTIC | Reads every `*.md`/`*.txt` in `docs/`; doc_id = filename | log line |
| 2 chunk | DETERMINISTIC | Splits on headings and blank lines, merges paragraphs up to ~600 chars; chunk_id = `<stem>_<i>`, keeps `doc_id` | log line |
| 3 index | DETERMINISTIC | fastembed `BAAI/bge-small-en-v1.5` → normalized numpy matrix, held in memory | log line |
| 4 retrieve | DETERMINISTIC | Cosine top-k (k=4) per question; written BEFORE any LLM call | `retrieval_results.json`: `question_id, question, retrieved_chunks[{doc_id, chunk_id, score, text}]` |
| 5 generate | LLM, ONE SEPARATE call per question, never batched | Prompt = template `prompts/answer.txt` + question + numbered chunks tagged with doc_id. The model returns JSON only: `{answer, citations[doc_id], supported, evidence[verbatim quotes]}`. Strip ```json fences; on a parse error, retry once with the error. Weak retrieval (top score < `MIN_SCORE`) → refuse with no LLM call. No API key or `--extractive` → answer with the best-matching sentence from the top chunk | `answers.json`: `question_id, answer, citations, supported, evidence, retrieved_doc_ids` |
| 6 validate | DETERMINISTIC | `support_check.py` rules below; fail-closed downgrade | `validation_report.json`: `summary{total, passed, failed, downgraded}`, `results[{question_id, passed, checks{}, errors[]}]` |

**Support-check rules (stage 6)**
1. Answer text is not empty.
2. `supported=true` ⇒ at least 1 citation.
3. Every citation is in the retrieved doc_ids.
4. Every evidence quote appears verbatim (whitespace-normalized) in a chunk from a cited doc.
5. `supported=false` ⇒ the answer is the refusal text "I can't answer that from the available documentation."

A supported answer that fails rule 2, 3 or 4 is rewritten to the refusal with `supported=false`, and the downgrade is recorded in the report.

## Feature coverage (mapped to the question)
- **MUST 1 Ingestion + retrieval**: stages 1–4 → `retrieval_results.json`
- **MUST 2 Grounded generation + refusal**: stage 5 → `answers.json`
- **MUST 3 Deterministic citation/support check**: stage 6 → `validation_report.json`
- **MUST 4 Interface**: `python app.py --question "..." [--extractive]` → JSON `{answer, citations, supported, sources[chunk_ids]}`, reusing the same functions
- **SHOULD 5 Tests**: `tests/test_rag.py` (pytest, no network):
  - chunk metadata is preserved
  - the password-reset question retrieves `auth_policy.md`
  - an off-topic question gets a refusal (through the threshold)
  - support_check rejects a citation that isn't in the retrieved set
- **SHOULD 6 Observability**: stdlib logging writes JSON lines to `logs/pipeline.jsonl`: docs loaded, chunks created, retrieval run, answer generated, validation passed or failed
- **STRETCH 7**: confidence threshold `MIN_SCORE` (env-overridable, calibrated on the sample corpus). Chosen because it cuts hallucination on off-topic questions and saves API quota. The separate prompt template also comes with it. The README explains why.

## Files
- **Code**
  - `llm.py`: OpenAI client; primary model → fallback model; backoff 2/4/8/16s on 429/5xx; cache at `cache/<prompt_hash>.json`; appends to `llm_calls.jsonl` (stage, question_id, UTC timestamp, provider, model, prompt_hash, input/output artifact, attempts, fallback_used); `load_dotenv(Path(__file__).resolve().parent / ".env")`
  - `retrieval.py`: `load_documents`, `chunk_documents`, `build_index`, `retrieve`
  - `generation.py`: `build_prompt`, `generate_answer` (LLM), `extractive_answer` (fallback)
  - `support_check.py`: `check_answer`, `enforce`
  - `pipeline.py`: runs the stages over `questions.json`, writes all artifacts plus `run_manifest.json` (stage start/end times, sha256 of every input file, models used); prints each stage name as it starts
  - `app.py`: the CLI
  - `validate.py`: output-file validator. Checks that files exist; schemas and field names are right; question ids line up across files; stage order is correct in the manifest; citations ⊂ retrieved; evidence quotes are verbatim; supported answers have citations; answers aren't empty. Prints PASS/FAIL and exits non-zero on any FAIL.
- **Data**
  - `docs/`: `auth_policy.md`, `api_rate_limits.md`, `kyc_verification.md`, `withdrawals.md`, `incident_escalation.md`
  - `questions.json`: 10 questions, 6 answerable and 4 unanswerable or partial; `expected_behavior` is `answerable` / `unanswerable`
  - `prompts/answer.txt`: the prompt template
- **Repo**
  - `TASK.md`: the question, verbatim
  - `PLAN.md`: this plan
  - `README.md`: architecture and layout, setup, how retrieval works, grounding and refusal, the stretch rationale, limitations and tradeoffs, done / not done
  - `Makefile` (`run`, `validate`, `test`), `requirements.txt` (openai, python-dotenv, fastembed, numpy, pytest), `.env.example` (key values left empty)
  - `.gitignore`: `.env`, `HANDOVER.md`, `cache/`, `__pycache__/`, `.venv/`, `venv/`

## Rules (from the question's technical constraints)
- Knowledge comes only from local docs, and no answers to the sample questions are hardcoded.
- It works when the docs and questions are swapped for equivalent files.
- Retrieval, prompting, generation and validation live in separate modules.
- Citations are never optional.
- Unsupported questions never get confident, made-up answers.
- The generation step can be replaced, and there is an extractive fallback for running without a key.
- API keys are never printed, logged or committed.
- Every function has a short docstring.

## Build order / git
1. Write TASK.md and PLAN.md, scaffold the repo, add docs and questions. Run `git init`, confirm `.env` is NOT staged in `git status`, commit, then `gh repo create support-rag-assistant --public --source=. --push`.
2. MUST: stages 1–6, `app.py`, `validate.py` → run, validate, push.
3. SHOULD: tests and logging → run, validate, push.
4. STRETCH: threshold calibration and README section → run, validate, push.

After each push, give a 2–3 line spoken summary.

## Verification
- `python pipeline.py` regenerates `retrieval_results.json`, `answers.json`, `validation_report.json` and `run_manifest.json`.
- `python validate.py` → every check PASS, exit 0.
- `python -m pytest -q` → green, no network.
- `python app.py --question "How long do withdrawals take?"` → a cited answer; an off-topic question → the refusal.
- `python pipeline.py --extractive` works without an API key.
