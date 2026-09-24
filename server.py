"""Minimal HTTP interface: POST /ask (JSON API) and GET / (a small web page). Same ask() as the CLI.

Usage: python server.py   ->  http://127.0.0.1:8000  (API docs at /docs)
"""
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import llm
from app import ROOT, ask, load_corpus

app = FastAPI(title="Support RAG Assistant")
OLLAMA_HELP = ("Install Ollama from https://ollama.com, run `ollama pull qwen2.5:1.5b`, "
               "make sure it is running, then try again. LM Studio works too (base URL http://localhost:1234/v1).")


class LLMConfig(BaseModel):
    """How answers are generated. env = .env keys; api = your key; local = Ollama/LM Studio; none = extractive."""
    mode: Literal["env", "api", "local", "none"]
    base_url: str | None = Field(default=None, max_length=300)
    model: str | None = Field(default=None, max_length=100)
    api_key: str | None = Field(default=None, max_length=300)


@app.get("/config")
def get_config() -> dict:
    """Active LLM backend and what .env provides (keys are never returned)."""
    return llm.status()


@app.post("/config")
def set_config(cfg: LLMConfig) -> dict:
    """Choose the LLM backend at runtime. The endpoint is tested before it is accepted; keys stay in memory only."""
    if cfg.mode == "env":
        if not llm.status()["env"]:
            raise HTTPException(400, "No API key is configured in .env. Enter a key, use a local LLM, or choose no LLM.")
        llm.set_override(None)
    elif cfg.mode == "none":
        llm.set_override([])
    else:
        if not (cfg.model or "").strip() or not (cfg.base_url or "").strip():
            raise HTTPException(422, "base_url and model are required")
        if cfg.mode == "api" and not (cfg.api_key or "").strip():
            raise HTTPException(422, "api_key is required")
        target = llm.make_target("local" if cfg.mode == "local" else "api", cfg.base_url.strip(),
                                 cfg.model.strip(), (cfg.api_key or "").strip() or None)
        error = llm.ping(target)
        if error:
            hint = f" {OLLAMA_HELP}" if cfg.mode == "local" else " Check the key, base URL and model name."
            raise HTTPException(400, f"Could not get a reply from {cfg.model} at {cfg.base_url}: {error.rstrip('.')}.{hint}")
        llm.set_override([target])
    return llm.status()


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
    mode: str | None = None
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
  * { box-sizing:border-box; } [hidden] { display:none !important; }
  body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }
  main { max-width:760px; margin:0 auto; padding:32px 16px; }
  h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:18px; margin:0 0 4px; }
  p.sub { color:var(--muted); margin:0 0 20px; }
  form.row { display:flex; gap:8px; flex-wrap:wrap; }
  input[type=text], input[type=password] { flex:1 1 320px; width:100%; padding:10px 12px; font:inherit;
          border:1px solid var(--line); border-radius:8px; background:var(--card); color:var(--fg); }
  button { padding:10px 18px; font:inherit; border:0; border-radius:8px; background:var(--accent); color:#fff; cursor:pointer; }
  button.link { background:none; color:var(--accent); padding:0; font-size:14px; text-decoration:underline; }
  button:disabled { opacity:.6; cursor:wait; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-top:20px; }
  .choice { display:block; border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin-top:10px; cursor:pointer; }
  .choice:has(input:checked) { border-color:var(--accent); }
  .choice:has(input:disabled) { opacity:.55; cursor:not-allowed; }
  .choice b { font-weight:600; } .choice .muted { display:block; margin-left:24px; }
  .fields { display:grid; gap:8px; margin:10px 0 0 24px; }
  .fields label { font-size:14px; color:var(--muted); }
  .badge { display:inline-block; font-size:13px; font-weight:600; padding:2px 10px; border-radius:99px; }
  .yes { color:var(--ok); background:var(--okbg); } .no { color:var(--no); background:var(--nobg); }
  .answer { font-size:18px; margin:10px 0; }
  .muted { color:var(--muted); font-size:14px; }
  code { font-size:13px; background:var(--bg); padding:1px 6px; border-radius:4px; }
  ul { padding-left:18px; margin:6px 0 0; } .err { color:var(--no); white-space:pre-wrap; }
  .status { margin:0 0 12px; }
</style></head>
<body><main>
  <h1>Support RAG Assistant</h1>
  <p class="sub">Answers only from the local <code>docs/</code> folder, with citations. Refuses when the docs don't support an answer.</p>

  <section id="setup" class="card" hidden aria-labelledby="setup-h">
    <h2 id="setup-h">How should answers be generated?</h2>
    <div class="muted">Retrieval always runs locally. Choose the model that writes the answer.</div>
    <form id="setupform">
      <label class="choice"><input type="radio" name="mode" value="env" id="opt-env"> <b>Server API key (.env)</b>
        <span class="muted" id="envdesc"></span></label>

      <label class="choice"><input type="radio" name="mode" value="api"> <b>Enter an API key</b>
        <span class="muted">Any OpenAI-compatible provider (Gemini, OpenAI, Groq…). Kept in server memory only, never saved.</span></label>
      <div class="fields" id="apifields" hidden>
        <label>Base URL <input type="text" id="apiurl" value="https://generativelanguage.googleapis.com/v1beta/openai/"></label>
        <label>Model <input type="text" id="apimodel" value="gemini-3.6-flash"></label>
        <label>API key <input type="password" id="apikey" autocomplete="off"></label>
      </div>

      <label class="choice"><input type="radio" name="mode" value="local"> <b>Local LLM (Ollama / LM Studio)</b>
        <span class="muted">Private and free, runs on this machine. Needs <a href="https://ollama.com" target="_blank" rel="noopener">Ollama</a> running with a model pulled, e.g. <code>ollama pull qwen2.5:1.5b</code>.</span></label>
      <div class="fields" id="localfields" hidden>
        <label>Base URL <input type="text" id="localurl" value="http://localhost:11434/v1"></label>
        <label>Model <input type="text" id="localmodel" value="qwen2.5:1.5b"></label>
      </div>

      <label class="choice"><input type="radio" name="mode" value="none"> <b>No LLM</b>
        <span class="muted">Extractive answers: the best-matching sentence from the docs. Works offline, less precise.</span></label>

      <div style="margin-top:14px"><button id="setupbtn">Continue</button></div>
      <div id="setuperr" class="err" role="alert" style="margin-top:10px"></div>
    </form>
  </section>

  <section id="askarea" hidden>
    <div class="muted status">Answering with: <code id="llmdesc"></code> · <button class="link" id="change" type="button">Change</button></div>
    <form id="f" class="row">
      <input type="text" id="q" placeholder="e.g. How long does a SEPA withdrawal take?" required aria-label="Question">
      <button id="b">Ask</button>
    </form>
    <div id="out" aria-live="polite"></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const describe = ts => ts.length ? ts.map(t => `${t.provider}/${t.model}`).join(' → ') : 'no LLM (extractive answers)';
const selected = () => (document.querySelector('input[name=mode]:checked') || {}).value;
const PREF = 'rag-llm-choice';  // remembers the last choice (never the key) so it's pre-selected next time
const loadPref = () => { try { return JSON.parse(localStorage.getItem(PREF) || 'null'); } catch { return null; } };
const savePref = p => { try { localStorage.setItem(PREF, JSON.stringify(p)); } catch {} };

function showFields() {
  $('apifields').hidden = selected() !== 'api';
  $('localfields').hidden = selected() !== 'local';
}

async function openSetup() {
  $('askarea').hidden = true; $('setup').hidden = false; $('setuperr').textContent = '';
  let s = {env: []};
  try { s = await (await fetch('/config')).json(); } catch { $('setuperr').textContent = 'Could not reach the server.'; }
  $('envdesc').textContent = s.env.length ? 'Configured: ' + describe(s.env) : 'No API key found in .env';
  $('opt-env').disabled = !s.env.length;
  const last = loadPref();
  if (last) ['apiurl', 'apimodel', 'localurl', 'localmodel'].forEach(k => { if (last[k]) $(k).value = last[k]; });
  let mode = last && last.mode;
  if (!mode || (mode === 'env' && !s.env.length)) mode = s.env.length ? 'env' : 'local';
  document.querySelector(`input[name=mode][value=${mode}]`).checked = true;
  showFields();
}

$('setupform').onchange = showFields;
$('setupform').onsubmit = async e => {
  e.preventDefault();
  const mode = selected(), btn = $('setupbtn');
  const body = {mode};
  if (mode === 'api') Object.assign(body, {base_url: $('apiurl').value, model: $('apimodel').value, api_key: $('apikey').value});
  if (mode === 'local') Object.assign(body, {base_url: $('localurl').value, model: $('localmodel').value});
  btn.disabled = true; btn.textContent = mode === 'api' || mode === 'local' ? 'Testing connection…' : 'Saving…';
  $('setuperr').textContent = '';
  try {
    const r = await fetch('/config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok) throw new Error(typeof d.detail === 'string' ? d.detail : d.detail.map(x => x.msg).join('; '));
    savePref({mode, apiurl: $('apiurl').value, apimodel: $('apimodel').value, localurl: $('localurl').value, localmodel: $('localmodel').value});
    $('apikey').value = '';
    $('llmdesc').textContent = describe(d.active);
    $('setup').hidden = true; $('askarea').hidden = false; $('q').focus();
  } catch (err) {
    $('setuperr').textContent = err.message;
  } finally { btn.disabled = false; btn.textContent = 'Continue'; }
};
$('change').onclick = openSetup;

$('f').onsubmit = async e => {
  e.preventDefault();
  const b = $('b'), out = $('out');
  b.disabled = true; b.textContent = 'Asking…';
  try {
    const r = await fetch('/ask', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question: $('q').value})});
    const d = await r.json();
    if (!r.ok) throw new Error(typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail));
    const v = d.validation;
    const fellBack = d.mode === 'extractive_fallback'
      ? '<div class="muted err" style="margin-top:6px">The LLM did not respond (quota, key or server down), so this is an extractive answer. Use “Change” to pick another backend.</div>' : '';
    out.innerHTML = `<div class="card">
      <span class="badge ${d.supported ? 'yes' : 'no'}">${d.supported ? 'Supported' : 'Not supported'}</span>
      <span class="muted">&nbsp;answered by: <code>${esc(d.mode || '')}</code></span>${fellBack}
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

openSetup();  // always ask on opening
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn
    load_corpus((ROOT / "docs").resolve())  # build the index before serving so the first request is fast
    uvicorn.run(app, host="127.0.0.1", port=8000)
