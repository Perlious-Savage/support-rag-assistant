# Support RAG Assistant

A small, grounded question-answering service for internal product and platform docs. It retrieves passages from local `docs/` and answers with citations. It refuses when the docs don't support an answer, and every answer goes through a deterministic check.

## Main flow (one command)
```bash
python pipeline.py          # or: make run
python validate.py          # or: make validate  -> PASS/FAIL per requirement
python -m pytest -q         # or: make test
python app.py --question "How long does a SEPA withdrawal take?"
python server.py            # or: make serve  -> web UI + API at http://127.0.0.1:8000
```

## Interfaces
Both return `answer`, `citations`, `supported` and `sources` (retrieved `doc_id`/`chunk_id`/`score`), plus the support-check `validation` result.
- **Web UI:** open http://127.0.0.1:8000. **When the page opens, it asks how answers should be generated:**
  - **Server API key (.env):** the Gemini primary → fallback models configured in `.env`
  - **Enter an API key:** any OpenAI-compatible provider (base URL + model + key). The key is kept in server memory only; it's never saved, logged or returned.
  - **Local LLM (Ollama / LM Studio):** e.g. `http://localhost:11434/v1` + `qwen2.5:1.5b`. No key needed, and nothing leaves the machine.
  - **No LLM:** extractive answers only (offline).

  The server sends a test prompt to the chosen backend before accepting it. If a local LLM isn't reachable, the page shows install steps (`ollama pull qwen2.5:1.5b`). "Change" reopens the choice at any time. Each answer shows which mode produced it (`llm`, `extractive`, `extractive_fallback`, `threshold`).
- **HTTP API:** `GET /config` shows the active backend (never keys); `POST /config` with `{"mode": "env"|"api"|"local"|"none", "base_url", "model", "api_key"}` switches it; `POST /ask` with `{"question": "...", "top_k": 4, "extractive": false}`. Interactive docs are at http://127.0.0.1:8000/docs.
  ```bash
  curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question":"How long do card refunds take?"}'
  ```
- **CLI:** `python app.py --question "..." [--extractive] [--top-k N] [--docs DIR]`.
No API key? `python pipeline.py --extractive` / `python app.py -q "..." --extractive` run fully offline.

## Setup
```bash
python -m pip install -r requirements.txt
cp .env.example .env        # fill LLM_API_KEY (+ optional FALLBACK_API_KEY); any OpenAI-compatible endpoint works
```
The first run downloads the local embedding model `BAAI/bge-small-en-v1.5` (~70 MB, via fastembed). After that, retrieval makes no API calls.

## Architecture
```
docs/*.md ──> load_documents ─> chunk ─> index ─> retrieve ─> generate ─> validate
                 (retrieval.py)                       │    (generation.py)  (support_check.py)
                                                      ▼          ▼                ▼
                                       retrieval_results.json  answers.json  validation_report.json
```
The stage order is enforced in `pipeline.py`, and each stage's timestamps are recorded in `run_manifest.json`.

| File | Responsibility |
|---|---|
| `retrieval.py` | Load docs, split into chunks (doc_id and chunk_id metadata), embed, cosine top-k |
| `prompts/answer.txt` | Prompt template, kept separate from code |
| `generation.py` | Build prompt, call LLM, normalize JSON, confidence-threshold refusal, extractive fallback |
| `llm.py` | Replaceable LLM client: primary → fallback model, backoff on 429/5xx, disk cache (`cache/`), call log |
| `support_check.py` | Deterministic citation/support rules; fail-closed downgrade to refusal |
| `pipeline.py` | Batch run over `questions.json`, writes all artifacts |
| `app.py` | CLI for one new question; `ask()` shared with the server |
| `server.py` | FastAPI: `POST /ask` + a one-page web UI at `/` |
| `validate.py` | Checks every artifact against the spec |
| `obs.py` | Structured JSON-lines logging (`logs/pipeline.jsonl`) |
| `tests/test_rag.py` | Chunk metadata, retrieval doc ids, refusal, support-check rules, `/ask` API |

Artifacts: `retrieval_results.json`, `answers.json`, `validation_report.json`, `run_manifest.json` (stage timings, sha256 of inputs, models), `llm_calls.jsonl` (one line per LLM call: stage, question_id, model, prompt_hash, attempts, fallback_used), `logs/pipeline.jsonl`.

## How retrieval works
- **Chunking:** documents are split on markdown headings, so each section becomes a chunk. A heading with no body (such as the doc title) is merged into the next section. Sections over 600 characters are split on paragraphs. Each chunk keeps its `doc_id` (the filename) and a `chunk_id` of the form `<stem>_<n>`.
- **Index:** chunks are embedded locally with `bge-small-en-v1.5` and stored as an L2-normalized numpy matrix.
- **Search:** cosine similarity (a dot product), top-4 per question.

## How grounding and refusal work
Four layers, cheapest first:
1. **Confidence threshold (stretch item).** If the best retrieval score is below `MIN_SCORE` (default 0.6, can be overridden with an environment variable), the system refuses **without calling the LLM**.
2. **Grounded prompt.** The model sees only the numbered, doc-tagged chunks. It must return JSON `{answer, citations, supported, evidence}`, with `evidence` quoted word for word from the context. It's told to set `supported:false` rather than guess. Anything not explicitly `supported: true` becomes the fixed refusal: *"I can't answer that from the available documentation."*
3. **Deterministic support check** (`support_check.py`), applied to every answer:
   - `answer_not_empty`
   - `supported_has_citation`: a supported answer has at least 1 citation
   - `citations_in_retrieved`: every cited doc was actually retrieved
   - `evidence_in_cited_chunks`: every evidence quote appears in the text of a cited chunk (after normalizing whitespace, case and markdown)
   - `unsupported_marked`: an unsupported answer is the refusal text
4. **Fail closed.** A supported answer that breaks any rule is replaced with the refusal and marked `downgraded` in `validation_report.json`.

## Why the confidence threshold (stretch item)
Off-topic questions are the most common way a RAG bot makes things up: the retriever always returns *something*, and the model tries to use it. On the sample corpus, answerable questions score ≥ 0.76 and a clearly off-topic question ("parental leave") scores 0.53, so 0.6 separates them cleanly. The threshold also saves an LLM call on every such question, which matters on a small free-tier quota. Borderline questions that pass the threshold are still handled by layers 2–4.

## Results on the sample set
- 10 questions: 6 answerable, 4 unanswerable or partly answerable.
- LLM mode: 10/10 support checks passed; 9/10 match the expected behaviour.
- The one mismatch is q9 ("exact Enterprise limits"). The model gave the grounded partial answer "custom limits agreed in the contract", which is supported by the doc, but the expected behaviour was a refusal.

## LLM backends and failover
`llm.py` tries the `.env` tiers in order: primary → fallback → optional `LOCAL_BASE_URL`/`LOCAL_MODEL` (Ollama, LM Studio, llama.cpp, no key). It retries 429/5xx with backoff (2/4/8/16 s) but skips retries once a *daily* quota is exhausted. If every tier fails, the answer falls back to extractive (`mode: extractive_fallback`) instead of an error. A backend chosen in the web UI replaces the `.env` tiers for that server process.

## Limitations and tradeoffs
- **Extractive fallback is crude.** It picks the sentence with the most question-word overlap, with no stemming. It refuses when overlap is weak (safe), but it can also "answer" with a sentence that only mentions the topic. Use LLM mode for real answers.
- **The evidence check proves the quotes exist, not that they imply the answer.** A model could quote real text and still draw a wrong conclusion from it. This check catches invented quotes and citations, not bad reasoning.
- **MIN_SCORE is calibrated on this corpus.** A different embedding model or corpus needs a new threshold. It can be set with the `MIN_SCORE` environment variable.
- **The LLM choice in the web UI is one per server process**, not per browser. That's fine for a local single-user tool. It also isn't cached, because the response cache is keyed by prompt only.
- **The server caches the index at startup**, so restart it after editing `docs/`.
- **The index is in memory and rebuilt every run.** That's fine for tens of docs. For thousands, persist the vectors (e.g. FAISS) and add BM25 for exact-term queries such as error codes.
- **The partly-answerable policy is binary.** The prompt forbids partial answers, but the model may still answer the documented part of a question (see q9).
- **Free-tier quota.** The client backs off (2/4/8/16 s) and then switches to the fallback model. Responses are cached by prompt hash, so reruns are instant and cost nothing.
