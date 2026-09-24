## Problem Statement
Build a small, runnable AI service that answers support-style questions from a local knowledge base.

The goal is to show practical AI engineering skill: document ingestion, retrieval, prompt construction, answer generation, and basic guardrails. Keep it simple, but production-minded.

Your service must:
- read a small set of local documents
- index them for retrieval
- accept a user question
- retrieve relevant passages
- generate an answer grounded in those passages
- return citations to the source documents
- refuse to answer when the documents do not support the claim

This is not a generic chatbot task. The evaluator will look for a clear separation between retrieval, prompting, generation, and validation.

Assume the use case is a lightweight internal help assistant for product or platform documentation, which is a realistic fit for an AI Engineer working on applied LLM systems.

---

## Input / Sample Data
Your solution must work from local files on disk.

Create and use a small sample corpus in:
- `docs/`

Use at least 5 short text or markdown documents. Keep them realistic and product-like. For example:
- account login and password reset
- API rate limits
- KYC / identity verification rules
- payment processing or withdrawal timing
- incident escalation policy

Also create a small evaluation set:
- `questions.json`

Include at least 8 questions total:
- 5 answerable from the docs
- 3 unanswerable or partially answerable

Example `questions.json` shape:
```json
[
  {
    "id": "q1",
    "question": "How many password reset attempts are allowed per hour?",
    "expected_behavior": "answerable"
  },
  {
    "id": "q2",
    "question": "Can support manually bypass identity verification for VIP users?",
    "expected_behavior": "unanswerable"
  }
]
```

The evaluator may replace your sample documents and questions with equivalent local files using the same structure. Your code must not depend on exact wording.

---

## MUST COMPLETE
### 1. Build a document ingestion and retrieval pipeline
Implement code that:
- loads documents from `docs/`
- splits them into chunks
- stores chunk metadata including document name
- creates a searchable index
- retrieves the top relevant chunks for a question

You may use embeddings, BM25, TF-IDF, or a simple hybrid approach. If an external model is used, the solution must still run from a clean checkout with clear setup steps.

Save retrieval outputs for the evaluation set to:
- `retrieval_results.json`

Each result should include at least:
```json
{
  "question_id": "q1",
  "retrieved_chunks": [
    {
      "doc_id": "auth_policy.md",
      "chunk_id": "auth_policy_3",
      "score": 0.91,
      "text": "..."
    }
  ]
}
```

### 2. Build an answer generation step with grounding
Implement an answer generation step that:
- takes a user question and retrieved context
- produces a concise answer
- includes citations to the supporting documents
- does not invent policy details not present in the retrieved text

If the answer is not supported by the retrieved context, the system must say so clearly.

Save outputs to:
- `answers.json`

Each answer record must include at least:
```json
{
  "question_id": "q1",
  "answer": "Users can request up to 3 password resets per hour.",
  "citations": ["auth_policy.md"],
  "supported": true
}
```

### 3. Add a simple citation / support check
Implement a deterministic validation step in code.

At minimum, check that:
- every supported answer includes at least one citation
- cited documents appear in the retrieved set
- unsupported answers are marked clearly
- answer text is not empty

You do not need to build a perfect hallucination detector. A simple, explicit support-checking rule is enough if it is implemented cleanly.

Save validation results to:
- `validation_report.json`

### 4. Expose the system through a small interface
Provide one of the following:
- a CLI such as `python app.py --question "..."`
- or a minimal API such as Flask/FastAPI with one `/ask` endpoint

The interface must return:
- answer
- citations
- whether the answer is supported
- retrieved sources or source IDs

---

## SHOULD ATTEMPT
### 5. Add basic tests
Include a few focused tests for the most important logic, such as:
- chunking or metadata preservation
- retrieval returning expected document IDs on sample data
- unsupported questions producing a refusal or uncertainty response

### 6. Add simple observability
Log the main pipeline stages, for example:
- documents loaded
- chunks created
- retrieval executed
- answer generated
- validation passed or failed

A simple structured log file is enough.

---

## STRETCH
### 7. Add one practical improvement
Choose one bounded improvement, such as:
- hybrid retrieval
- reranking
- duplicate chunk filtering
- prompt templates separated from code
- answer length controls
- a confidence threshold for refusing weak retrievals

Document why you chose it.

---

## Required Artifacts or Expected Outcome
Your repository must include:
- `docs/`
- `questions.json`
- `retrieval_results.json`
- `answers.json`
- `validation_report.json`
- runnable source code
- `README.md`

If you add tests, include them in the repository.

---

## Validation Requirements
The evaluator should be able to:
- install dependencies
- run your pipeline from a clean checkout
- regenerate the output artifacts
- ask at least one new question through the CLI or API

Include a single command in the README for the main flow, for example:
```bash
python run_pipeline.py
```
or:
```bash
make run
```

---

## Deliverable Format
Keep the solution small and practical.

Include a short README that explains:
- architecture and file layout
- setup steps
- how retrieval works
- how grounding / refusal works
- limitations and tradeoffs

---

## Tools
Python is required.

Suggested libraries:
- `fastapi` or `flask`
- `pydantic`
- `pytest`
- `scikit-learn`
- `sentence-transformers`
- `faiss-cpu` if you prefer vector search

You may use an LLM provider or a local model, but your code should make the generation step clearly replaceable. If external API keys are needed, provide a fallback path for local testing, such as returning extractive answers from retrieved chunks.

---

## Technical Constraints
- Use only local input documents for knowledge.
- Do not hardcode answers to the sample questions.
- Keep retrieval, generation, and validation as separate steps in code.
- Do not treat citations as optional.
- Unsupported questions must not receive confident fabricated answers.
- The solution must be runnable and understandable within the time limit.
- Favor clarity and correct boundaries over heavy infrastructure.
