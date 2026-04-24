"""
BanglaLLM Web Interface
Run : pip install flask sentencepiece transformers torch
Then: python app.py
Open: http://localhost:5000
"""

import torch
import sentencepiece as spm
from flask import Flask, request, jsonify, render_template_string
from transformers import LlamaForCausalLM, LlamaConfig

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
FINAL_MODEL_DIR = "./final_model"
SP_MODEL_PATH   = "./tokenizer/bangla_spm.model"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
# LOAD MODEL + TOKENIZER ONCE AT STARTUP
# ─────────────────────────────────────────────────────────────
print("Loading SentencePiece tokenizer...")
sp = spm.SentencePieceProcessor(model_file=SP_MODEL_PATH)

print(f"Loading model from {FINAL_MODEL_DIR}...")
config = LlamaConfig.from_pretrained(FINAL_MODEL_DIR)
model  = LlamaForCausalLM.from_pretrained(
    FINAL_MODEL_DIR,
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
)
model.to(DEVICE)
model.eval()

PARAM_COUNT = sum(p.numel() for p in model.parameters()) / 1e6
VOCAB_SIZE  = sp.get_piece_size()
print(f"Model ready | {PARAM_COUNT:.1f}M params | device: {DEVICE}")

# ─────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────
def run_generate(prompt, max_new_tokens=150, temperature=0.8, top_p=0.9, top_k=50):
    ids       = [sp.bos_id()] + sp.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens     = max_new_tokens,
            do_sample          = temperature > 0,
            temperature        = temperature,
            top_p              = top_p,
            top_k              = top_k,
            eos_token_id       = sp.eos_id(),
            pad_token_id       = 0,
            repetition_penalty = 1.2,
        )

    new_ids = output[0][len(ids):].tolist()
    return sp.decode(new_ids)

# ─────────────────────────────────────────────────────────────
# HTML  (fully self-contained, no external files needed)
# ─────────────────────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="bn">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>বাংলা LLM</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+Bengali:wght@400;600;700&family=Tiro+Bangla&display=swap" rel="stylesheet">
<style>
:root {
  --bg:      #0d0f14;
  --surface: #151820;
  --border:  #252a35;
  --accent:  #e8b86d;
  --accent2: #7eb8c9;
  --text:    #e8e4dc;
  --muted:   #6b7385;
  --r:       12px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Noto Serif Bengali', serif;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 48px 20px 80px;
}

.wrap { width: 100%; max-width: 760px; }

/* ── Header ── */
header { text-align: center; margin-bottom: 48px; }

.orb {
  width: 52px; height: 52px;
  border-radius: 50%;
  background: radial-gradient(circle at 35% 35%, #e8b86d44, transparent 70%),
              radial-gradient(circle at 65% 65%, #7eb8c922, transparent 70%);
  border: 1.5px solid #e8b86d55;
  margin: 0 auto 18px;
  box-shadow: 0 0 32px #e8b86d22;
}

h1 {
  font-family: 'Tiro Bangla', serif;
  font-size: clamp(1.9rem, 5vw, 2.8rem);
  font-weight: 700;
  background: linear-gradient(135deg, var(--accent) 0%, #f0d4a0 50%, var(--accent2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.15;
}

.tagline {
  margin-top: 8px;
  color: var(--muted);
  font-size: 0.82rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 14px;
  padding: 5px 14px;
  border-radius: 20px;
  border: 1px solid var(--border);
  background: var(--surface);
  font-size: 0.78rem;
  color: var(--muted);
}
.badge b { color: var(--accent); font-weight: 600; }

/* ── Card ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 26px;
  margin-bottom: 18px;
}

.field-label {
  display: block;
  font-size: 0.78rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 10px;
}

textarea {
  width: 100%;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  font-family: 'Noto Serif Bengali', serif;
  font-size: 1.05rem;
  line-height: 1.75;
  padding: 13px 15px;
  resize: vertical;
  min-height: 96px;
  outline: none;
  transition: border-color 0.2s;
}
textarea:focus { border-color: var(--accent); }
textarea::placeholder { color: var(--muted); }

/* ── Sliders ── */
.sliders {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
  margin-top: 20px;
}

.sl-group { display: flex; flex-direction: column; gap: 6px; }

.sl-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

input[type=range] {
  flex: 1;
  accent-color: var(--accent);
  cursor: pointer;
}

.sl-val {
  font-size: 0.82rem;
  color: var(--accent);
  min-width: 34px;
  text-align: right;
}

/* ── Button ── */
.btn-row {
  margin-top: 22px;
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 12px;
}

.hint { font-size: 0.75rem; color: var(--muted); }

button.primary {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--accent);
  color: #0d0f14;
  border: none;
  border-radius: 8px;
  padding: 11px 26px;
  font-family: 'Noto Serif Bengali', serif;
  font-size: 0.95rem;
  font-weight: 700;
  cursor: pointer;
  transition: opacity 0.15s, transform 0.1s;
}
button.primary:hover   { opacity: 0.85; }
button.primary:active  { transform: scale(0.98); }
button.primary:disabled { opacity: 0.35; cursor: not-allowed; }

/* ── Spinner ── */
.spin {
  width: 16px; height: 16px;
  border: 2px solid rgba(13,15,20,0.25);
  border-top-color: #0d0f14;
  border-radius: 50%;
  animation: spin 0.65s linear infinite;
  display: none;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Error ── */
.err {
  display: none;
  margin-top: 12px;
  padding: 10px 14px;
  background: rgba(220,80,80,0.08);
  border: 1px solid rgba(220,80,80,0.25);
  border-radius: 8px;
  font-size: 0.86rem;
  color: #e07878;
}

/* ── Output ── */
.out-card { display: none; }
.out-card.show { display: block; }

.out-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.meta {
  display: flex;
  gap: 14px;
  font-size: 0.76rem;
  color: var(--muted);
}
.meta span b { color: var(--accent2); }

button.copy {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--muted);
  padding: 4px 12px;
  font-size: 0.76rem;
  cursor: pointer;
  transition: all 0.2s;
}
button.copy:hover { border-color: var(--accent); color: var(--accent); opacity: 1; }

.out-text {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  font-size: 1.05rem;
  line-height: 1.85;
  white-space: pre-wrap;
  word-break: break-word;
}

.prompt-part { color: var(--accent); font-weight: 600; }
.gen-part    { color: var(--text); }

/* ── Footer ── */
footer {
  margin-top: 44px;
  text-align: center;
  font-size: 0.75rem;
  color: var(--muted);
  letter-spacing: 0.05em;
}

@media (max-width: 540px) {
  .sliders { grid-template-columns: 1fr; }
  .hint    { display: none; }
}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="orb"></div>
    <h1>বাংলা ভাষা মডেল</h1>
    <p class="tagline">Bengali Language Model — Trained from Scratch</p>
    <div class="badge">
      LLaMA&thinsp;3 architecture &nbsp;·&nbsp;
      <b>{{ params }}</b> params &nbsp;·&nbsp;
      vocab <b>{{ vocab }}</b> &nbsp;·&nbsp;
      <b>{{ device }}</b>
    </div>
  </header>

  <!-- Input -->
  <div class="card">
    <label class="field-label" for="prompt">বাংলায় লিখুন — Write your Bengali prompt</label>
    <textarea id="prompt" rows="4"
      placeholder="যেমন: বাংলাদেশের ইতিহাস হলো...&#10;(Ctrl+Enter to generate)"></textarea>

    <div class="sliders">
      <div class="sl-group">
        <label class="field-label">সর্বোচ্চ টোকেন</label>
        <div class="sl-row">
          <input type="range" id="max-tokens" min="20" max="300" value="150"
                 oninput="V('max-val', this.value)">
          <span class="sl-val" id="max-val">150</span>
        </div>
      </div>
      <div class="sl-group">
        <label class="field-label">Temperature</label>
        <div class="sl-row">
          <input type="range" id="temperature" min="0.1" max="1.5" step="0.05" value="0.8"
                 oninput="V('temp-val', parseFloat(this.value).toFixed(2))">
          <span class="sl-val" id="temp-val">0.80</span>
        </div>
      </div>
      <div class="sl-group">
        <label class="field-label">Top-P</label>
        <div class="sl-row">
          <input type="range" id="top-p" min="0.5" max="1.0" step="0.05" value="0.9"
                 oninput="V('topp-val', parseFloat(this.value).toFixed(2))">
          <span class="sl-val" id="topp-val">0.90</span>
        </div>
      </div>
    </div>

    <div class="btn-row">
      <span class="hint">Ctrl + Enter</span>
      <button class="primary" id="gen-btn" onclick="generate()">
        <div class="spin" id="spinner"></div>
        <span id="btn-text">তৈরি করুন →</span>
      </button>
    </div>
    <div class="err" id="err"></div>
  </div>

  <!-- Output -->
  <div class="card out-card" id="out-card">
    <div class="out-header">
      <div class="meta">
        <span>ইনপুট <b id="tok-in">—</b> টোকেন</span>
        <span>আউটপুট <b id="tok-out">—</b> টোকেন</span>
      </div>
      <button class="copy" onclick="copyOut()">কপি করুন</button>
    </div>
    <div class="out-text" id="out-text"></div>
  </div>

  <footer>BanglaLLM &nbsp;·&nbsp; Built with LLaMA 3 architecture &nbsp;·&nbsp; SentencePiece Unigram tokenizer</footer>
</div>

<script>
function V(id, val) { document.getElementById(id).textContent = val; }

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function generate() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;

  const btn     = document.getElementById('gen-btn');
  const spinner = document.getElementById('spinner');
  const btnText = document.getElementById('btn-text');
  const errDiv  = document.getElementById('err');
  const outCard = document.getElementById('out-card');

  btn.disabled          = true;
  spinner.style.display = 'block';
  btnText.textContent   = 'তৈরি হচ্ছে...';
  errDiv.style.display  = 'none';
  outCard.classList.remove('show');

  try {
    const res = await fetch('/generate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        prompt:         prompt,
        max_new_tokens: parseInt(document.getElementById('max-tokens').value),
        temperature:    parseFloat(document.getElementById('temperature').value),
        top_p:          parseFloat(document.getElementById('top-p').value),
      })
    });

    const data = await res.json();

    if (data.error) {
      errDiv.textContent   = 'ত্রুটি: ' + data.error;
      errDiv.style.display = 'block';
      return;
    }

    document.getElementById('tok-in').textContent  = data.tokens_in;
    document.getElementById('tok-out').textContent = data.tokens_out;
    document.getElementById('out-text').innerHTML  =
      '<span class="prompt-part">' + esc(prompt) + '</span>' +
      '<span class="gen-part">'   + esc(data.generated) + '</span>';

    outCard.classList.add('show');
    outCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  } catch (e) {
    errDiv.textContent   = 'সার্ভার সংযোগ ব্যর্থ: ' + e.message;
    errDiv.style.display = 'block';
  } finally {
    btn.disabled          = false;
    spinner.style.display = 'none';
    btnText.textContent   = 'তৈরি করুন →';
  }
}

function copyOut() {
  const text = document.getElementById('out-text').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const b = document.querySelector('.copy');
    b.textContent = 'কপি হয়েছে ✓';
    setTimeout(() => b.textContent = 'কপি করুন', 2000);
  });
}

document.getElementById('prompt').addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') generate();
});
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(
        HTML,
        params = f"{PARAM_COUNT:.1f}M",
        vocab  = f"{VOCAB_SIZE:,}",
        device = DEVICE.upper(),
    )

@app.route("/generate", methods=["POST"])
def generate_route():
    data = request.get_json(force=True)

    prompt         = data.get("prompt", "").strip()
    max_new_tokens = max(10,  min(int(data.get("max_new_tokens", 150)), 512))
    temperature    = max(0.1, min(float(data.get("temperature",   0.8)), 2.0))
    top_p          = max(0.1, min(float(data.get("top_p",         0.9)), 1.0))

    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    try:
        ids_in    = [sp.bos_id()] + sp.encode(prompt)
        generated = run_generate(prompt, max_new_tokens, temperature, top_p)
        ids_out   = sp.encode(generated)

        return jsonify({
            "generated"  : generated,
            "tokens_in"  : len(ids_in),
            "tokens_out" : len(ids_out),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/info")
def info():
    return jsonify({
        "parameters" : f"{PARAM_COUNT:.1f}M",
        "vocab_size" : VOCAB_SIZE,
        "device"     : DEVICE,
    })

if __name__ == "__main__":
    print(f"\n{'─'*50}")
    print(f"  BanglaLLM  →  http://localhost:5000")
    print(f"{'─'*50}\n")
    app.run(host="0.0.0.0", port=5000, debug=False)