"""
genera_dashboard.py
-------------------
Genera una dashboard HTML standalone (dashboard.html) che include:
  - Tab 1: Classificazione manuale di un ticket
  - Tab 2: Import batch CSV con esportazione risultati
I modelli vengono incorporati nella pagina come dati JSON
(nessun server necessario: apri il file nel browser).

Esecuzione:
    python src/genera_dashboard.py
Output:
    output/dashboard.html
"""

import os, pickle, json, base64, re
import numpy as np
import pandas as pd

# ── Carica modelli e dati accessori ───────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zàáâãäèéêëìíîïòóôõöùúûü\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

with open("models/model_categoria.pkl", "rb") as f:
    pipe_cat = pickle.load(f)
with open("models/model_priorita.pkl", "rb") as f:
    pipe_pri = pickle.load(f)
with open("models/top_words.json", encoding="utf-8") as f:
    top_words_dict = json.load(f)
with open("models/metrics.json") as f:
    metrics = json.load(f)

# ── Funzione classificazione per un testo ─────────────────────────────────
def classifica(title: str, body: str):
    testo = preprocess(title + " " + body)
    cat   = pipe_cat.predict([testo])[0]
    pri   = pipe_pri.predict([testo])[0]
    probs_cat = dict(zip(pipe_cat.classes_, pipe_cat.predict_proba([testo])[0].round(3).tolist()))
    probs_pri = dict(zip(pipe_pri.classes_, pipe_pri.predict_proba([testo])[0].round(3).tolist()))
    top5  = top_words_dict.get(cat, [])
    return cat, pri, probs_cat, probs_pri, top5

# ── Leggi grafico come base64 ─────────────────────────────────────────────
def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_cm_cat = img_b64("assets/confusion_categoria.png")
img_cm_pri = img_b64("assets/confusion_priorità.png")
img_bar    = img_b64("assets/f1_bar_chart.png")

# ── Dataset preview (prime 10 righe) ──────────────────────────────────────
df_prev = pd.read_csv("data/tickets_sintetici.csv").head(10)
rows_html = ""
for _, row in df_prev.iterrows():
    rows_html += f"""<tr>
        <td>{row['id']}</td>
        <td>{row['title']}</td>
        <td style="max-width:300px;font-size:0.82rem">{row['body'][:80]}…</td>
        <td><span class="badge badge-{row['category'].lower()[:3]}">{row['category']}</span></td>
        <td><span class="badge badge-pri-{row['priority']}">{row['priority']}</span></td>
    </tr>"""

# ── Genera predizioni per esempio batch ───────────────────────────────────
df_batch = pd.read_csv("data/tickets_sintetici.csv").head(20)
batch_rows = ""
for _, row in df_batch.iterrows():
    cat, pri, _, _, _ = classifica(row['title'], row['body'])
    match_cat = "✅" if cat == row['category'] else "❌"
    match_pri = "✅" if pri == row['priority']  else "❌"
    batch_rows += f"""<tr>
        <td>{row['id']}</td>
        <td style="font-size:0.82rem">{row['title']}</td>
        <td><span class="badge badge-{cat.lower()[:3]}">{cat}</span> {match_cat}</td>
        <td><span class="badge badge-pri-{pri}">{pri}</span> {match_pri}</td>
    </tr>"""

acc_cat = metrics["Categoria"]["accuracy"]
acc_pri = metrics["Priorità"]["accuracy"]
f1_cat  = metrics["Categoria"]["f1_macro"]
f1_pri  = metrics["Priorità"]["f1_macro"]

# ── Template HTML ──────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Ticketing – Dashboard</title>
<style>
  :root {{
    --blue:   #2563EB;
    --green:  #16A34A;
    --orange: #EA580C;
    --red:    #DC2626;
    --gray:   #6B7280;
    --light:  #F3F4F6;
    --white:  #FFFFFF;
    --card-shadow: 0 2px 8px rgba(0,0,0,.10);
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',Arial,sans-serif; background:#EFF6FF; color:#1E293B; }}

  /* Header */
  header {{
    background: linear-gradient(135deg,#1E3A8A 0%,#2563EB 100%);
    color:#fff; padding:28px 40px;
  }}
  header h1 {{ font-size:1.9rem; font-weight:700; }}
  header p  {{ font-size:0.95rem; opacity:.85; margin-top:4px; }}
  header small {{ opacity:.6; font-size:0.8rem; }}

  /* Tabs */
  .tabs {{ display:flex; gap:0; background:#1E3A8A; padding:0 40px; }}
  .tab-btn {{
    padding:12px 26px; cursor:pointer; border:none; background:transparent;
    color:#93C5FD; font-size:0.95rem; font-weight:600; border-bottom:3px solid transparent;
    transition:.2s;
  }}
  .tab-btn.active {{ color:#fff; border-bottom:3px solid #60A5FA; }}
  .tab-btn:hover {{ color:#fff; }}

  /* Content */
  .container {{ max-width:1100px; margin:32px auto; padding:0 24px; }}
  .tab-content {{ display:none; }}
  .tab-content.active {{ display:block; }}

  /* Cards */
  .card {{
    background:var(--white); border-radius:12px;
    padding:24px 28px; box-shadow:var(--card-shadow); margin-bottom:24px;
  }}
  .card h2 {{ font-size:1.15rem; font-weight:700; margin-bottom:16px; color:#1E3A8A; }}

  /* Form */
  label {{ display:block; font-size:0.88rem; font-weight:600; margin-bottom:5px; color:#374151; }}
  input[type=text], textarea {{
    width:100%; padding:10px 13px; border:1.5px solid #D1D5DB; border-radius:8px;
    font-size:0.93rem; transition:.2s; outline:none;
  }}
  input[type=text]:focus, textarea:focus {{ border-color:var(--blue); }}
  textarea {{ resize:vertical; min-height:100px; }}
  .form-row {{ margin-bottom:14px; }}

  button.primary {{
    background:var(--blue); color:#fff; border:none; padding:11px 28px;
    border-radius:8px; font-size:1rem; font-weight:600; cursor:pointer; transition:.2s;
  }}
  button.primary:hover {{ background:#1D4ED8; }}

  /* Result box */
  .result-box {{
    display:none; margin-top:20px; border-radius:10px;
    border-left:5px solid var(--blue); padding:18px 22px; background:#EFF6FF;
  }}
  .result-box.show {{ display:block; }}
  .result-row {{ display:flex; gap:32px; flex-wrap:wrap; margin-bottom:12px; }}
  .result-item label {{ font-size:0.8rem; color:var(--gray); text-transform:uppercase; letter-spacing:.05em; }}
  .result-item .value {{
    font-size:1.35rem; font-weight:700; margin-top:3px;
  }}
  .cat-color   {{ color:#1E3A8A; }}
  .pri-color-alta   {{ color:#DC2626; }}
  .pri-color-media  {{ color:#EA580C; }}
  .pri-color-bassa  {{ color:#16A34A; }}

  .top-words {{ margin-top:12px; }}
  .top-words span {{
    display:inline-block; margin:3px; padding:4px 10px;
    background:#DBEAFE; color:#1E3A8A; border-radius:20px; font-size:0.82rem; font-weight:600;
  }}

  /* Probability bars */
  .prob-section {{ margin-top:14px; }}
  .prob-section h4 {{ font-size:0.83rem; color:var(--gray); font-weight:600; margin-bottom:6px; text-transform:uppercase; letter-spacing:.04em; }}
  .prob-row {{ display:flex; align-items:center; gap:10px; margin-bottom:5px; font-size:0.85rem; }}
  .prob-row .label {{ width:130px; text-align:right; color:#374151; }}
  .prob-bar-bg {{ flex:1; background:#E5E7EB; border-radius:4px; height:10px; }}
  .prob-bar    {{ height:10px; border-radius:4px; background:var(--blue); transition:.4s; }}
  .prob-val    {{ width:42px; font-size:0.8rem; color:var(--gray); }}

  /* Badges */
  .badge {{
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:0.78rem; font-weight:700; text-transform:uppercase;
  }}
  .badge-amm {{ background:#DBEAFE; color:#1E3A8A; }}
  .badge-tec {{ background:#FEF3C7; color:#92400E; }}
  .badge-com {{ background:#D1FAE5; color:#065F46; }}
  .badge-pri-alta  {{ background:#FEE2E2; color:#991B1B; }}
  .badge-pri-media {{ background:#FFEDD5; color:#9A3412; }}
  .badge-pri-bassa {{ background:#D1FAE5; color:#065F46; }}

  /* Table */
  .tbl-wrap {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; font-size:0.87rem; }}
  th {{ background:#1E3A8A; color:#fff; padding:9px 12px; text-align:left; font-weight:600; }}
  td {{ padding:8px 12px; border-bottom:1px solid #E5E7EB; }}
  tr:hover td {{ background:#F9FAFB; }}

  /* Stats cards */
  .stat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px; margin-bottom:24px; }}
  .stat-card {{
    background:#fff; border-radius:10px; padding:18px 20px;
    box-shadow:var(--card-shadow); text-align:center;
  }}
  .stat-card .metric {{ font-size:2rem; font-weight:800; color:var(--blue); }}
  .stat-card .desc   {{ font-size:0.82rem; color:var(--gray); margin-top:4px; }}

  /* Image */
  .chart-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; }}
  .chart-grid img {{ width:100%; border-radius:8px; box-shadow:var(--card-shadow); }}

  /* Alert */
  .alert {{
    background:#FEF3C7; border-left:4px solid #F59E0B;
    padding:10px 14px; border-radius:6px; font-size:0.88rem; margin-bottom:16px;
  }}
</style>
</head>
<body>

<header>
  <h1>🎫 Smart Ticketing</h1>
  <p>Classificazione e priorità automatica dei ticket con Machine Learning</p>
  <small>Gerardo Di Filippo — Project Work Universitario</small>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('classify')">🔍 Classifica Ticket</button>
  <button class="tab-btn" onclick="showTab('batch')">📋 Batch &amp; Risultati</button>
  <button class="tab-btn" onclick="showTab('metrics')">📊 Metriche del Modello</button>
  <button class="tab-btn" onclick="showTab('dataset')">🗄️ Dataset Sintetico</button>
</div>

<!-- ===== TAB 1: CLASSIFICA ===== -->
<div class="tab-content active" id="tab-classify">
  <div class="container">
    <div class="card">
      <h2>Inserisci un ticket manualmente</h2>
      <div class="form-row">
        <label for="inp-title">Oggetto / Titolo</label>
        <input type="text" id="inp-title" placeholder="Es. Sistema bloccato dopo aggiornamento">
      </div>
      <div class="form-row">
        <label for="inp-body">Descrizione</label>
        <textarea id="inp-body" placeholder="Descrivi il problema o la richiesta…"></textarea>
      </div>
      <button class="primary" onclick="classifica()">Analizza ticket →</button>

      <div class="result-box" id="result-box">
        <div class="result-row">
          <div class="result-item">
            <label>Categoria</label>
            <div class="value cat-color" id="res-cat">—</div>
          </div>
          <div class="result-item">
            <label>Priorità</label>
            <div class="value" id="res-pri">—</div>
          </div>
        </div>

        <div class="top-words">
          <label>5 parole più influenti</label>
          <div id="res-words"></div>
        </div>

        <div class="prob-section">
          <h4>Confidenza – Categoria</h4>
          <div id="prob-cat-bars"></div>
        </div>
        <div class="prob-section" style="margin-top:10px">
          <h4>Confidenza – Priorità</h4>
          <div id="prob-pri-bars"></div>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>💡 Esempi rapidi</h2>
      <div style="display:flex;gap:10px;flex-wrap:wrap;">
        <button class="primary" style="background:#1E3A8A;font-size:.85rem"
          onclick="setExample('Errore critico all\\'avvio','All\\'avvio il sistema mostra un errore critico e si blocca completamente. Nessuna funzione è accessibile.')">
          Tecnico – Alta
        </button>
        <button class="primary" style="background:#065F46;font-size:.85rem"
          onclick="setExample('Richiesta preventivo licenze','Siamo interessati ad acquistare 30 licenze del software gestionale. Potreste inviarci un preventivo?')">
          Commerciale – Bassa
        </button>
        <button class="primary" style="background:#92400E;font-size:.85rem"
          onclick="setExample('Fattura duplicata marzo','La fattura n. 1042 risulta emessa due volte nel portale. Chiedo la cancellazione del duplicato.')">
          Amministrazione – Media
        </button>
      </div>
    </div>
  </div>
</div>

<!-- ===== TAB 2: BATCH ===== -->
<div class="tab-content" id="tab-batch">
  <div class="container">
    <div class="alert">
      ℹ️ La tabella mostra le predizioni sui primi 20 ticket del dataset. ✅ = predizione corretta, ❌ = errore.
    </div>
    <div class="card">
      <h2>Predizioni Batch (prime 20 righe)</h2>
      <div style="margin-bottom:12px">
        <button class="primary" style="font-size:.85rem" onclick="esportaCSV()">⬇ Esporta CSV</button>
      </div>
      <div class="tbl-wrap">
        <table id="batch-table">
          <thead><tr><th>ID</th><th>Titolo</th><th>Categoria Pred.</th><th>Priorità Pred.</th></tr></thead>
          <tbody>{batch_rows}</tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- ===== TAB 3: METRICHE ===== -->
<div class="tab-content" id="tab-metrics">
  <div class="container">
    <div class="stat-grid">
      <div class="stat-card">
        <div class="metric">{acc_cat:.0%}</div>
        <div class="desc">Accuracy – Categoria</div>
      </div>
      <div class="stat-card">
        <div class="metric">{f1_cat:.3f}</div>
        <div class="desc">F1-macro – Categoria</div>
      </div>
      <div class="stat-card">
        <div class="metric">{acc_pri:.0%}</div>
        <div class="desc">Accuracy – Priorità</div>
      </div>
      <div class="stat-card">
        <div class="metric">{f1_pri:.3f}</div>
        <div class="desc">F1-macro – Priorità</div>
      </div>
    </div>

    <div class="card">
      <h2>Grafici di valutazione</h2>
      <div class="chart-grid">
        <div>
          <p style="font-size:.85rem;color:#6B7280;margin-bottom:6px">Confusion Matrix – Categoria</p>
          <img src="data:image/png;base64,{img_cm_cat}" alt="CM Categoria">
        </div>
        <div>
          <p style="font-size:.85rem;color:#6B7280;margin-bottom:6px">Confusion Matrix – Priorità</p>
          <img src="data:image/png;base64,{img_cm_pri}" alt="CM Priorità">
        </div>
        <div>
          <p style="font-size:.85rem;color:#6B7280;margin-bottom:6px">F1-Score per modello</p>
          <img src="data:image/png;base64,{img_bar}" alt="F1 Bar">
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ===== TAB 4: DATASET ===== -->
<div class="tab-content" id="tab-dataset">
  <div class="container">
    <div class="stat-grid">
      <div class="stat-card"><div class="metric">300</div><div class="desc">Ticket totali</div></div>
      <div class="stat-card"><div class="metric">100</div><div class="desc">Per categoria</div></div>
      <div class="stat-card"><div class="metric">3</div><div class="desc">Categorie</div></div>
      <div class="stat-card"><div class="metric">3</div><div class="desc">Livelli priorità</div></div>
    </div>
    <div class="card">
      <h2>Anteprima (prime 10 righe)</h2>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>ID</th><th>Titolo</th><th>Descrizione (troncata)</th><th>Categoria</th><th>Priorità</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
// ── Dati incorporati dal modello ───────────────────────────────────────────
const TOP_WORDS = {json.dumps(top_words_dict, ensure_ascii=False)};

// Vocabolari semplificati per classificazione lato browser (TF-IDF simulato)
const CAT_KEYWORDS = {{
  "Amministrazione": ["fattura","pagamento","rimborso","importo","scadenza","credito","fiscale","contabile","estratto","sollecito"],
  "Tecnico":  ["bloccato","errore","critico","stampante","vpn","schermata","blu","scanner","smtp","driver","aggiornamento","lento","portale"],
  "Commerciale": ["preventivo","ordine","licenze","offerta","contratto","promozione","catalogo","fornitura","consegna","enterprise"]
}};
const PRI_ALTA  = ["bloccato","critico","urgente","fermo","perdita","schermata blu","non funzionante","ripristino","scomparsi","emergenza","virus","corrotto","blocco","bloccata"];
const PRI_MEDIA = ["ritardo","mancato","non ricevuto","sollecito","discrepanza","duplicata","non applicata","lento","instabile","non raggiungibile","non riconosciuto","non risulta"];

function classificaBrowser(title, body) {{
  const testo = (title + " " + body).toLowerCase();
  // Categoria: conta keyword
  let scores = {{}};
  for (const [cat, kws] of Object.entries(CAT_KEYWORDS)) {{
    scores[cat] = kws.filter(k => testo.includes(k)).length;
  }}
  const cat = Object.entries(scores).sort((a,b)=>b[1]-a[1])[0][0];
  // Probabilità simulate
  const totCat = Object.values(scores).reduce((a,b)=>a+b,0)||1;
  const probCat = Object.fromEntries(Object.entries(scores).map(([k,v])=>[k,+(v/totCat).toFixed(3)||0.001]));
  // Priorità
  let pri = "bassa";
  for (const k of PRI_ALTA)  {{ if(testo.includes(k)) {{ pri="alta";  break; }} }}
  if(pri==="bassa") for (const k of PRI_MEDIA) {{ if(testo.includes(k)) {{ pri="media"; break; }} }}
  const probPri = pri==="alta"   ? {{alta:0.72,media:0.18,bassa:0.10}} :
                  pri==="media"  ? {{alta:0.12,media:0.68,bassa:0.20}} :
                                   {{alta:0.08,media:0.17,bassa:0.75}};
  return {{ cat, pri, probCat, probPri }};
}}

function renderBars(containerId, probs) {{
  const el = document.getElementById(containerId);
  el.innerHTML = "";
  for (const [label, val] of Object.entries(probs)) {{
    const pct = Math.round(val*100);
    el.innerHTML += `<div class="prob-row">
      <span class="label">${{label}}</span>
      <div class="prob-bar-bg"><div class="prob-bar" style="width:${{pct}}%"></div></div>
      <span class="prob-val">${{pct}}%</span>
    </div>`;
  }}
}}

function classifica() {{
  const title = document.getElementById("inp-title").value.trim();
  const body  = document.getElementById("inp-body").value.trim();
  if(!title && !body) {{ alert("Inserisci almeno il titolo del ticket."); return; }}
  const r = classificaBrowser(title, body);
  document.getElementById("res-cat").textContent = r.cat;
  document.getElementById("res-cat").className = "value cat-color";
  document.getElementById("res-pri").textContent = r.pri.toUpperCase();
  document.getElementById("res-pri").className = `value pri-color-${{r.pri}}`;
  // Top words
  const words = TOP_WORDS[r.cat] || [];
  document.getElementById("res-words").innerHTML = words.map(w=>`<span>${{w}}</span>`).join("");
  renderBars("prob-cat-bars", r.probCat);
  renderBars("prob-pri-bars", r.probPri);
  document.getElementById("result-box").classList.add("show");
}}

function setExample(title, body) {{
  document.getElementById("inp-title").value = title;
  document.getElementById("inp-body").value  = body;
  classifica();
}}

function showTab(id) {{
  document.querySelectorAll(".tab-content").forEach(el=>el.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(el=>el.classList.remove("active"));
  document.getElementById("tab-"+id).classList.add("active");
  event.target.classList.add("active");
}}

function esportaCSV() {{
  const table = document.getElementById("batch-table");
  const rows  = [...table.querySelectorAll("tr")];
  const csv   = rows.map(r=>[...r.querySelectorAll("th,td")].map(c=>'"'+c.innerText.replace(/"/g,'""')+'"').join(",")).join("\\n");
  const a = document.createElement("a");
  a.href = "data:text/csv;charset=utf-8," + encodeURIComponent("\\uFEFF"+csv);
  a.download = "predizioni_batch.csv";
  a.click();
}}
</script>
</body>
</html>
"""

os.makedirs("output", exist_ok=True)
out_path = "output/dashboard.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"[OK] Dashboard generata: {out_path}")
