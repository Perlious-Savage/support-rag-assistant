"""Minimal HTTP interface: POST /ask (JSON API) and GET / (a small web page). Same ask() as the CLI.

Usage: python server.py   ->  http://127.0.0.1:8000  (API docs at /docs)
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app import ROOT, ask, load_corpus

app = FastAPI(title="Support RAG Assistant")


class AskRequest(BaseModel):
    """Request body for /ask."""
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)
    extractive: bool = False


class Source(BaseModel):
    """One retrieved chunk reference."""
    doc_id: str
    chunk_id: str
    score: float


class AskResponse(BaseModel):
    """Answer with citations, support flag and retrieved sources."""
    question: str
    answer: str
    citations: list[str]
    supported: bool
    sources: list[Source]
    validation: dict


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> dict:
    """Answer one question from docs/ with citations; refuses when unsupported."""
    if not req.question.strip():
        raise HTTPException(422, "question must not be blank")
    try:
        return ask(req.question.strip(), top_k=req.top_k, extractive=req.extractive)
    except RuntimeError as e:  # all LLM endpoints failed
        raise HTTPException(503, f"LLM unavailable: {e}. Retry, or set extractive=true.")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the single-page UI."""
    return PAGE


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Support RAG Assistant</title>
<style>
  :root { --bg:#f7f7f5; --card:#fff; --fg:#1d1d1b; --muted:#6b6b66; --line:#e2e2dc; --accent:#2f5bd3;
          --ok:#1f7a45; --okbg:#e6f4ea; --no:#9a3412; --nobg:#fdeee4; }
  @media (prefers-color-scheme: dark) { :root { --bg:#161615; --card:#1f1f1d; --fg:#ececea; --muted:#a3a39c;
          --line:#33332f; --accent:#7ea2ff; --ok:#7fd4a0; --okbg:#1b3325; --no:#f5a878; --nobg:#3a2519; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }
  main { max-width:760px; margin:0 auto; padding:32px 16px; }
  h1 { font-size:22px; margin:0 0 4px; } p.sub { color:var(--muted); margin:0 0 20px; }
  form { display:flex; gap:8px; flex-wrap:wrap; }
  input[type=text] { flex:1 1 320px; padding:10px 12px; font:inherit; border:1px solid var(--line);
          border-radius:8px; background:var(--card); color:var(--fg); }
  button { padding:10px 18px; font:inherit; border:0; border-radius:8px; background:var(--accent); color:#fff; cursor:pointer; }
  button:disabled { opacity:.6; cursor:wait; }
  label.opt { color:var(--muted); font-size:14px; display:flex; align-items:center; gap:6px; width:100%; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-top:20px; }
  .badge { display:inline-block; font-size:13px; font-weight:600; padding:2px 10px; border-radius:99px; }
  .yes { color:var(--ok); background:var(--okbg); } .no { color:var(--no); background:var(--nobg); }
  .answer { font-size:18px; margin:10px 0; }
  .muted { color:var(--muted); font-size:14px; }
  code { font-size:13px; background:var(--bg); padding:1px 6px; border-radius:4px; }
  ul { padding-left:18px; margin:6px 0 0; } .err { color:var(--no); }
</style></head>
<body><main>
  <h1>Support RAG Assistant</h1>
  <p class="sub">Answers only from the local <code>docs/</code> folder, with citations. Refuses when the docs don't support an answer.</p>
  <form id="f">
    <input type="text" id="q" placeholder="e.g. How long does a SEPA withdrawal take?" required autofocus aria-label="Question">
    <button id="b">Ask</button>
    <label class="opt"><input type="checkbox" id="x"> Extractive mode (no LLM)</label>
  </form>
  <div id="out" aria-live="polite"></div>
</main>
<script>
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  const b = document.getElementById('b'), out = document.getElementById('out');
  b.disabled = true; b.textContent = 'Asking…';
  try {
    const r = await fetch('/ask', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question: document.getElementById('q').value, extractive: document.getElementById('x').checked})});
    const d = await r.json();
    if (!r.ok) throw new Error(typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail));
    const v = d.validation;
    out.innerHTML = `<div class="card">
      <span class="badge ${d.supported ? 'yes' : 'no'}">${d.supported ? 'Supported' : 'Not supported'}</span>
      <div class="answer">${esc(d.answer)}</div>
      <div class="muted">Citations: ${d.citations.length ? d.citations.map(c => `<code>${esc(c)}</code>`).join(' ') : 'none'}</div>
      <div class="muted" style="margin-top:10px">Retrieved sources:</div>
      <ul class="muted">${d.sources.map(s => `<li><code>${esc(s.chunk_id)}</code> (${esc(s.doc_id)}), score ${s.score.toFixed(3)}</li>`).join('')}</ul>
      <div class="muted" style="margin-top:10px">Support check: ${v.passed ? 'passed' : 'failed → downgraded to refusal (' + esc(v.errors.join(', ')) + ')'}</div>
    </div>`;
  } catch (err) {
    out.innerHTML = `<div class="card err">Error: ${esc(err.message)}</div>`;
  } finally { b.disabled = false; b.textContent = 'Ask'; }
};
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn
    load_corpus((ROOT / "docs").resolve())  # build the index before serving so the first request is fast
    uvicorn.run(app, host="127.0.0.1", port=8000)
