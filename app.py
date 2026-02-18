"""
Portfolio Optimizer — Web App Server
=====================================
Avvia con:  python app.py
Apri sul telefono: http://TUO-SERVER:8080
"""

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_file

# ── Setup path ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "portfolio-optimizer-secret-2024")

# ── Directory di lavoro ────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "output"
STATE_FILE  = BASE_DIR / "portfolio_state.json"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Stato dei job in esecuzione ────────────────────────────────
# (ogni analisi gira in background e aggiorna questo dizionario)
jobs: dict = {}   # job_id → {status, progress, message, result, error}


# ═══════════════════════════════════════════════════════════════
# BACKGROUND WORKER — esegue l'analisi senza bloccare il browser
# ═══════════════════════════════════════════════════════════════

def run_analysis_job(job_id: str, mode: str, capital: float,
                     max_stocks: int, tickers: list):
    """
    Gira in un thread separato.
    Aggiorna jobs[job_id] con progress e risultati.
    """
    def upd(pct, msg):
        jobs[job_id].update({"progress": pct, "message": msg})

    try:
        jobs[job_id] = {"status": "running", "progress": 0,
                         "message": "Avvio analisi...", "result": None, "error": None}

        from utils.config import Config
        from utils.logger import setup_logger
        from data.downloader import DataDownloader
        from analysis.fundamentals import FundamentalAnalyzer
        from analysis.technical import TechnicalAnalyzer
        from analysis.news import NewsAnalyzer
        from analysis.scorer import StockScorer
        from optimization.portfolio import PortfolioOptimizer
        from output.report import ReportGenerator

        config = Config()
        logger = setup_logger()

        # ── Carica config ─────────────────────────────────────
        ticker_list = tickers if tickers else config.DEFAULT_TICKERS

        # ── 1. Download ───────────────────────────────────────
        upd(5, "📥 Download dati di mercato (può richiedere 1-2 minuti)...")
        dl = DataDownloader(period="5y", refresh=False)
        price_data, ticker_info = dl.download_all(ticker_list)
        if price_data.empty:
            raise Exception("Impossibile scaricare i dati. Verifica la connessione.")
        valid = [t for t in ticker_list if t in price_data.columns]
        upd(20, f"✓ {len(valid)} ticker scaricati")

        # ── 2. Tecnica ────────────────────────────────────────
        upd(30, "📊 Analisi tecnica in corso...")
        tech = TechnicalAnalyzer(price_data).analyze_all(valid)
        upd(40, f"✓ Analisi tecnica completata")

        # ── 3. Fondamentale ───────────────────────────────────
        upd(50, "🔍 Analisi fondamentale (P/E, ROE, FCF)...")
        fund = FundamentalAnalyzer(ticker_info).analyze_all(valid)
        upd(58, "✓ Analisi fondamentale completata")

        # ── 4. Pre-selezione ──────────────────────────────────
        upd(62, "⭐ Pre-selezione candidati...")
        pre_scorer = StockScorer(tech, fund, ticker_info)
        pre_ranked = pre_scorer.rank_stocks()
        pool = pre_ranked.head(min(max_stocks * 2, len(pre_ranked)))["ticker"].tolist()

        # ── 5. Notizie ────────────────────────────────────────
        upd(68, "📰 Analisi notizie e sentiment...")
        news = NewsAnalyzer(ticker_info).analyze_tickers(pool)
        upd(76, "✓ Notizie analizzate")

        # ── 6. Scoring + Ottimizzazione ───────────────────────
        upd(80, "⚖️  Scoring finale e ottimizzazione portafoglio...")
        scorer   = StockScorer(tech, fund, ticker_info, news)
        ranked   = scorer.rank_stocks()
        selected = scorer.select_top_stocks(ranked, min_stocks=6,
                                             max_stocks=max_stocks)
        sel_prices = price_data[selected]
        opt        = PortfolioOptimizer(price_data=sel_prices,
                                        scores=ranked, capital=capital)
        portfolio  = opt.optimize()
        backtest   = opt.backtest(portfolio)
        upd(90, f"✓ {len(selected)} titoli selezionati")

        # ── 7. Report HTML ────────────────────────────────────
        upd(95, "📄 Generazione report HTML...")
        reporter = ReportGenerator(
            portfolio=portfolio, backtest=backtest,
            price_data=price_data, ticker_info=ticker_info,
            scores=ranked, capital=capital,
        )
        reporter.save_report()

        # ── 8. Salva stato ────────────────────────────────────
        _save_state(portfolio, capital, ticker_info)

        upd(100, "✅ Analisi completata!")
        jobs[job_id].update({
            "status":  "done",
            "result": {
                "tickers":   selected,
                "return":    f"{portfolio['expected_return']:.1%}",
                "sharpe":    f"{portfolio['sharpe_ratio']:.2f}",
                "vol":       f"{portfolio['expected_volatility']:.1%}",
                "review":    portfolio.get("next_review", "—"),
                "n_stocks":  len(selected),
                "capital":   capital,
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
            }
        })

    except Exception as e:
        jobs[job_id].update({
            "status":  "error",
            "error":   str(e),
            "message": f"❌ Errore: {e}",
        })


def run_update_job(job_id: str):
    """Aggiorna solo i prezzi correnti (veloce ~15 sec)."""
    def upd(pct, msg):
        jobs[job_id].update({"progress": pct, "message": msg})

    try:
        jobs[job_id] = {"status": "running", "progress": 0,
                         "message": "Aggiornamento prezzi...", "result": None, "error": None}

        state = _load_state()
        if not state:
            raise Exception("Nessun portafoglio salvato. Prima esegui un'analisi completa.")

        upd(10, "📥 Download prezzi correnti...")
        import yfinance as yf
        tickers = list(state.get("holdings", {}).keys())
        if not tickers:
            raise Exception("Portafoglio vuoto nel file di stato.")

        prices_now = {}
        for t in tickers:
            try:
                d = yf.download(t, period="2d", auto_adjust=True, progress=False)
                if not d.empty:
                    prices_now[t] = float(d["Close"].iloc[-1].squeeze())
            except Exception:
                pass

        upd(80, "📊 Calcolo P&L...")
        holdings = state.get("holdings", {})
        pnl = []
        total_cost  = 0
        total_value = 0
        for t, h in holdings.items():
            curr = prices_now.get(t)
            if curr is None:
                continue
            avg   = h.get("avg_price", curr)
            qty   = h.get("quantity", 0)
            cost  = avg * qty
            value = curr * qty
            gain  = value - cost
            pnl.append({
                "ticker": t,
                "qty":    round(qty, 3),
                "avg":    round(avg, 2),
                "curr":   round(curr, 2),
                "value":  round(value, 0),
                "gain":   round(gain, 0),
                "gain_pct": round((curr / avg - 1) * 100, 1) if avg > 0 else 0,
            })
            total_cost  += cost
            total_value += value

        state["last_update"] = datetime.now().strftime("%d/%m/%Y %H:%M")
        state["total_value"] = round(total_value, 0)
        state["total_gain"]  = round(total_value - total_cost, 0)
        _save_state_raw(state)

        upd(100, "✅ Prezzi aggiornati!")
        jobs[job_id].update({
            "status": "done",
            "result": {
                "pnl":         pnl,
                "total_value": round(total_value, 0),
                "total_gain":  round(total_value - total_cost, 0),
                "total_gain_pct": round((total_value / total_cost - 1) * 100, 1)
                                  if total_cost > 0 else 0,
                "timestamp":   datetime.now().strftime("%d/%m/%Y %H:%M"),
            }
        })

    except Exception as e:
        jobs[job_id].update({
            "status": "error",
            "error":  str(e),
            "message": f"❌ Errore: {e}",
        })


# ═══════════════════════════════════════════════════════════════
# STATE HELPERS
# ═══════════════════════════════════════════════════════════════

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}

def _save_state_raw(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))

def _save_state(portfolio: dict, capital: float, ticker_info: dict):
    state = _load_state()
    holdings = {}
    for alloc in portfolio.get("allocation", []):
        t = alloc["ticker"]
        holdings[t] = {
            "weight":    round(alloc.get("weight", 0), 4),
            "amount":    round(alloc.get("amount_eur", 0), 0),
            "quantity":  0,   # non calcolato qui, aggiornato manualmente
            "avg_price": 0,
        }
    state.update({
        "last_analysis":    datetime.now().strftime("%d/%m/%Y %H:%M"),
        "capital":          capital,
        "expected_return":  round(portfolio.get("expected_return", 0), 4),
        "sharpe":           round(portfolio.get("sharpe_ratio", 0), 4),
        "next_review":      portfolio.get("next_review", "—"),
        "tickers":          portfolio.get("tickers", []),
        "holdings":         {**state.get("holdings", {}), **holdings},
    })
    _save_state_raw(state)


# ═══════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/api/start-analysis", methods=["POST"])
def api_start_analysis():
    """Avvia l'analisi completa in background."""
    data      = request.get_json() or {}
    capital   = float(data.get("capital", 10000))
    max_stocks= int(data.get("max_stocks", 12))
    tickers   = data.get("tickers", [])
    job_id    = str(uuid.uuid4())[:8]
    t = threading.Thread(
        target=run_analysis_job,
        args=(job_id, "full", capital, max_stocks, tickers),
        daemon=True
    )
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/start-update", methods=["POST"])
def api_start_update():
    """Avvia aggiornamento prezzi in background."""
    job_id = str(uuid.uuid4())[:8]
    t = threading.Thread(target=run_update_job, args=(job_id,), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>")
def api_job_status(job_id):
    """Polling: restituisce stato del job."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


@app.route("/api/state")
def api_state():
    """Stato corrente del portafoglio salvato."""
    return jsonify(_load_state())


@app.route("/api/report")
def api_report():
    """Scarica il report HTML generato."""
    path = OUTPUT_DIR / "portfolio_report.html"
    if not path.exists():
        return jsonify({"error": "Nessun report disponibile"}), 404
    return send_file(str(path))


@app.route("/api/update-holding", methods=["POST"])
def api_update_holding():
    """Aggiorna quantità e prezzo medio di un titolo nel portafoglio."""
    data   = request.get_json() or {}
    ticker = data.get("ticker")
    qty    = data.get("quantity")
    avg    = data.get("avg_price")
    if not ticker:
        return jsonify({"error": "ticker mancante"}), 400
    state = _load_state()
    if "holdings" not in state:
        state["holdings"] = {}
    if ticker not in state["holdings"]:
        state["holdings"][ticker] = {}
    if qty  is not None: state["holdings"][ticker]["quantity"]  = float(qty)
    if avg  is not None: state["holdings"][ticker]["avg_price"] = float(avg)
    _save_state_raw(state)
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════
# MAIN PAGE — interfaccia mobile
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(_MOBILE_UI)


# ═══════════════════════════════════════════════════════════════
# MOBILE UI — unica pagina HTML/CSS/JS responsive
# ═══════════════════════════════════════════════════════════════

_MOBILE_UI = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#060d1a">
<title>Portfolio Optimizer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
:root{
  --bg:#060d1a;--bg1:#0a1628;--bg2:#0f172a;
  --border:#1e293b;--text:#e2e8f0;--muted:#64748b;
  --blue:#60A5FA;--green:#34D399;--yellow:#FBBF24;
  --red:#F87171;--purple:#818CF8;
  --radius:14px;--safe-bottom:env(safe-area-inset-bottom,0px)
}
html,body{background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text',sans-serif;
  font-size:15px;min-height:100vh;overscroll-behavior:none}

/* ── NAV BAR ── */
.nav{position:fixed;bottom:0;left:0;right:0;
  background:rgba(6,13,26,.95);backdrop-filter:blur(20px);
  border-top:1px solid var(--border);
  display:flex;justify-content:space-around;align-items:center;
  padding:10px 0 calc(10px + var(--safe-bottom));z-index:100}
.nav-btn{display:flex;flex-direction:column;align-items:center;gap:4px;
  color:var(--muted);font-size:10px;font-weight:600;letter-spacing:.03em;
  background:none;border:none;cursor:pointer;padding:4px 16px;
  transition:color .2s;min-width:60px}
.nav-btn.active{color:var(--blue)}
.nav-btn svg{width:22px;height:22px}

/* ── HEADER ── */
.header{padding:16px 20px 8px;
  background:linear-gradient(180deg,rgba(6,13,26,1) 0%,rgba(6,13,26,0) 100%);
  position:sticky;top:0;z-index:50}
.header h1{font-size:20px;font-weight:800;
  background:linear-gradient(90deg,var(--blue),var(--purple));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header .sub{font-size:12px;color:var(--muted);margin-top:2px}

/* ── SCREEN ── */
.screen{display:none;padding:0 16px calc(90px + var(--safe-bottom));padding-top:8px}
.screen.active{display:block}
.content{padding-bottom:8px}

/* ── CARD ── */
.card{background:var(--bg2);border:1px solid var(--border);
  border-radius:var(--radius);padding:18px;margin-bottom:14px}
.card-title{font-size:11px;color:var(--muted);text-transform:uppercase;
  letter-spacing:.08em;font-weight:700;margin-bottom:14px}

/* ── KPI ── */
.kpi-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}
.kpi-row.three{grid-template-columns:1fr 1fr 1fr}
.kpi{background:var(--bg1);border:1px solid var(--border);
  border-radius:10px;padding:14px 12px}
.kpi .val{font-size:22px;font-weight:800;font-family:'SF Mono',monospace;
  letter-spacing:-1px;line-height:1}
.kpi .lbl{font-size:10px;color:var(--muted);margin-top:5px;font-weight:600}

/* ── BUTTONS ── */
.btn{display:block;width:100%;padding:16px;border-radius:12px;
  border:none;font-size:15px;font-weight:700;cursor:pointer;
  transition:opacity .15s,transform .1s;letter-spacing:-.01em}
.btn:active{opacity:.8;transform:scale(.98)}
.btn-primary{background:linear-gradient(135deg,#1d4ed8,#7c3aed);color:#fff}
.btn-green{background:#065f46;color:#34d399;border:1px solid #064e3b}
.btn-outline{background:transparent;color:var(--blue);
  border:1px solid #1e40af;margin-top:8px}
.btn-small{padding:10px 16px;font-size:13px;border-radius:9px}
.btn-red{background:#450a0a;color:#f87171;border:1px solid #7f1d1d}

/* ── FORM ── */
.form-group{margin-bottom:14px}
.form-label{font-size:12px;color:var(--muted);font-weight:600;
  display:block;margin-bottom:6px;letter-spacing:.02em}
.form-input{width:100%;padding:13px 14px;background:var(--bg1);
  border:1px solid var(--border);border-radius:10px;
  color:var(--text);font-size:15px;outline:none;appearance:none}
.form-input:focus{border-color:var(--blue)}

/* ── PROGRESS ── */
.progress-wrap{display:none;padding:20px 0}
.progress-bar-bg{background:var(--bg1);border-radius:8px;height:8px;overflow:hidden}
.progress-bar-fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--purple));
  border-radius:8px;transition:width .4s ease;width:0%}
.progress-msg{font-size:13px;color:var(--muted);margin-top:10px;min-height:20px;
  text-align:center}
.progress-pct{font-size:28px;font-weight:800;color:var(--blue);
  text-align:center;margin-bottom:8px;font-family:monospace}

/* ── TICKER LIST ── */
.ticker-row{display:flex;align-items:center;justify-content:space-between;
  padding:12px 0;border-bottom:1px solid var(--border)}
.ticker-row:last-child{border-bottom:none}
.ticker-name{font-weight:700;font-size:14px}
.ticker-sub{font-size:11px;color:var(--muted);margin-top:2px}
.badge{padding:4px 9px;border-radius:6px;font-size:11px;font-weight:700}
.badge-green{background:#052e16;color:#4ade80;border:1px solid #14532d}
.badge-red{background:#450a0a;color:#f87171;border:1px solid #7f1d1d}
.badge-blue{background:#0c1a3a;color:#60a5fa;border:1px solid #1e3a8a}
.badge-muted{background:var(--bg1);color:var(--muted);border:1px solid var(--border)}

/* ── ALERT ── */
.alert{padding:14px;border-radius:10px;font-size:13px;line-height:1.5;margin-bottom:14px}
.alert-info{background:#0c1a3a;border:1px solid #1e3a8a;color:#93c5fd}
.alert-success{background:#052e16;border:1px solid #14532d;color:#4ade80}
.alert-error{background:#450a0a;border:1px solid #7f1d1d;color:#f87171}
.alert-warn{background:#451a03;border:1px solid #7c2d12;color:#fdba74}

/* ── REPORT FRAME ── */
.report-frame{width:100%;height:calc(100vh - 140px);border:none;
  border-radius:var(--radius);background:var(--bg2)}

/* ── PNL TABLE ── */
.pnl-row{display:grid;grid-template-columns:1fr auto;
  align-items:center;gap:8px;padding:11px 0;
  border-bottom:1px solid var(--border)}
.pnl-row:last-child{border-bottom:none}

/* ── EMPTY STATE ── */
.empty{text-align:center;padding:50px 20px;color:var(--muted)}
.empty-icon{font-size:48px;margin-bottom:16px}
.empty h3{font-size:17px;font-weight:700;color:var(--text);margin-bottom:8px}
.empty p{font-size:13px;line-height:1.6}

/* ── SECTION TITLE ── */
.section-title{font-size:12px;color:var(--muted);font-weight:700;
  text-transform:uppercase;letter-spacing:.08em;
  margin:20px 0 10px;padding-left:2px}

/* ── LOADING SPINNER ── */
@keyframes spin{to{transform:rotate(360deg)}}
.spinner{width:20px;height:20px;border:2px solid var(--border);
  border-top-color:var(--blue);border-radius:50%;
  animation:spin .7s linear infinite;display:inline-block;vertical-align:middle}

/* ── MODAL ── */
.modal-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);
  z-index:200;align-items:flex-end;justify-content:center}
.modal-bg.open{display:flex}
.modal{background:var(--bg2);border:1px solid var(--border);
  border-radius:var(--radius) var(--radius) 0 0;
  padding:20px 20px calc(20px + var(--safe-bottom));width:100%;
  max-height:85vh;overflow-y:auto}
.modal-title{font-size:16px;font-weight:800;margin-bottom:16px}
.modal-close{float:right;background:none;border:none;color:var(--muted);
  font-size:20px;cursor:pointer}
</style>
</head>
<body>

<!-- ══ HEADER ══════════════════════════════════════════════════ -->
<div class="header">
  <h1>📈 Portfolio Optimizer</h1>
  <div class="sub" id="headerSub">Caricamento...</div>
</div>

<!-- ══ SCREEN: DASHBOARD ═══════════════════════════════════════ -->
<div class="screen active" id="screen-home">
<div class="content">

  <!-- Stato portafoglio -->
  <div id="dashContent">
    <div class="empty">
      <div class="empty-icon">📊</div>
      <h3>Nessun portafoglio</h3>
      <p>Vai su <b>Analisi</b> per creare<br>il tuo primo portafoglio ottimizzato</p>
    </div>
  </div>

</div>
</div>

<!-- ══ SCREEN: ANALISI ═════════════════════════════════════════ -->
<div class="screen" id="screen-analisi">
<div class="content">

  <div class="section-title">Parametri</div>
  <div class="card">
    <div class="form-group">
      <label class="form-label">💰 Capitale da investire (€)</label>
      <input class="form-input" type="number" id="inputCapital"
             value="10000" min="1000" step="500">
    </div>
    <div class="form-group">
      <label class="form-label">📦 Numero massimo di titoli</label>
      <input class="form-input" type="number" id="inputMaxStocks"
             value="12" min="5" max="20">
    </div>
    <div class="form-group">
      <label class="form-label">🔍 Ticker specifici (opzionale)
        <span style="color:var(--muted);font-weight:400"> — lascia vuoto per usare l'universo completo</span>
      </label>
      <input class="form-input" type="text" id="inputTickers"
             placeholder="es: AAPL MSFT GOOGL (separati da spazio)">
    </div>
  </div>

  <!-- Progress -->
  <div class="progress-wrap" id="analysisProgress">
    <div class="progress-pct" id="progressPct">0%</div>
    <div class="progress-bar-bg">
      <div class="progress-bar-fill" id="progressFill"></div>
    </div>
    <div class="progress-msg" id="progressMsg">Avvio...</div>
  </div>

  <!-- Buttons -->
  <button class="btn btn-primary" id="btnAnalisi" onclick="startAnalysis()">
    🚀 Avvia Analisi Completa
  </button>
  <button class="btn btn-green btn-small" style="margin-top:10px" onclick="startUpdate()">
    🔄 Aggiorna solo prezzi (veloce)
  </button>

  <div class="alert alert-info" style="margin-top:16px">
    ⏱️ L'analisi completa richiede <b>5-15 minuti</b> — puoi minimizzare Safari, ti avviserà al termine
  </div>

</div>
</div>

<!-- ══ SCREEN: PORTAFOGLIO ═════════════════════════════════════ -->
<div class="screen" id="screen-portfolio">
<div class="content">
  <div id="portfolioContent">
    <div class="empty">
      <div class="empty-icon">💼</div>
      <h3>Portafoglio vuoto</h3>
      <p>Esegui un'analisi per vedere i titoli consigliati</p>
    </div>
  </div>
</div>
</div>

<!-- ══ SCREEN: REPORT ══════════════════════════════════════════ -->
<div class="screen" id="screen-report">
<div class="content" style="padding:0 0 90px">
  <div id="reportContent">
    <div class="empty" style="margin-top:40px">
      <div class="empty-icon">📄</div>
      <h3>Nessun report</h3>
      <p>Esegui un'analisi completa per generare il report interattivo</p>
      <button class="btn btn-outline btn-small" style="margin-top:20px;width:auto;padding:10px 24px"
              onclick="showScreen('analisi')">Vai ad Analisi</button>
    </div>
  </div>
</div>
</div>

<!-- ══ SCREEN: IMPOSTAZIONI ═══════════════════════════════════ -->
<div class="screen" id="screen-settings">
<div class="content">

  <div class="section-title">Il mio portafoglio reale</div>
  <div class="card">
    <div class="card-title">Inserisci le quantità che hai comprato</div>
    <p style="font-size:12px;color:var(--muted);margin-bottom:14px">
      Inserisci la quantità di ogni titolo che possiedi e il prezzo a cui l'hai comprato.
      Serve per calcolare il tuo P&L reale.
    </p>
    <div id="holdingsList"></div>
    <button class="btn btn-outline btn-small" style="margin-top:10px" onclick="openAddHolding()">
      ＋ Aggiungi titolo manualmente
    </button>
  </div>

  <div class="section-title">Info</div>
  <div class="card">
    <div id="settingsInfo" style="font-size:13px;color:var(--muted);line-height:1.8">
      Caricamento...
    </div>
  </div>

</div>
</div>

<!-- ══ NAV BAR ═════════════════════════════════════════════════ -->
<nav class="nav">
  <button class="nav-btn active" id="nav-home" onclick="showScreen('home')">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
      <polyline points="9 22 9 12 15 12 15 22"/>
    </svg>
    Dashboard
  </button>
  <button class="nav-btn" id="nav-analisi" onclick="showScreen('analisi')">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
    </svg>
    Analisi
  </button>
  <button class="nav-btn" id="nav-portfolio" onclick="showScreen('portfolio')">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <rect x="2" y="7" width="20" height="14" rx="2"/>
      <path d="M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2"/>
    </svg>
    Portafoglio
  </button>
  <button class="nav-btn" id="nav-report" onclick="showScreen('report')">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
      <line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
    </svg>
    Report
  </button>
  <button class="nav-btn" id="nav-settings" onclick="showScreen('settings')">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="3"/>
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
    </svg>
    Impostazioni
  </button>
</nav>

<!-- ══ MODAL: aggiungi titolo ══════════════════════════════════ -->
<div class="modal-bg" id="modalHolding">
  <div class="modal">
    <button class="modal-close" onclick="closeModal()">✕</button>
    <div class="modal-title">Aggiungi titolo</div>
    <div class="form-group">
      <label class="form-label">Ticker (es: AAPL)</label>
      <input class="form-input" id="mTicker" placeholder="AAPL" style="text-transform:uppercase">
    </div>
    <div class="form-group">
      <label class="form-label">Quantità (numero azioni)</label>
      <input class="form-input" id="mQty" type="number" placeholder="10" step="0.001">
    </div>
    <div class="form-group">
      <label class="form-label">Prezzo medio di acquisto (€/$)</label>
      <input class="form-input" id="mAvg" type="number" placeholder="150.00" step="0.01">
    </div>
    <button class="btn btn-primary" onclick="saveHolding()">Salva</button>
  </div>
</div>

<script>
// ══ STATE ══════════════════════════════════════════════════════
let currentScreen = 'home';
let pollingTimer  = null;
let appState      = {};

// ══ NAVIGATION ═════════════════════════════════════════════════
function showScreen(name){
  document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('screen-'+name).classList.add('active');
  document.getElementById('nav-'+name).classList.add('active');
  currentScreen = name;
  if(name==='home')      loadDashboard();
  if(name==='portfolio') loadPortfolio();
  if(name==='report')    loadReport();
  if(name==='settings')  loadSettings();
}

// ══ DASHBOARD ══════════════════════════════════════════════════
async function loadDashboard(){
  const r = await fetch('/api/state');
  appState = await r.json();
  const el = document.getElementById('dashContent');

  if(!appState.tickers || !appState.tickers.length){
    el.innerHTML = `<div class="empty">
      <div class="empty-icon">📊</div>
      <h3>Nessun portafoglio</h3>
      <p>Vai su <b>Analisi</b> per creare<br>il tuo primo portafoglio ottimizzato</p>
    </div>`;
    document.getElementById('headerSub').textContent = 'Pronto';
    return;
  }

  const gain    = appState.total_gain || 0;
  const gainPct = appState.total_gain_pct || 0;
  const gainColor = gain >= 0 ? 'var(--green)' : 'var(--red)';
  const gainSign  = gain >= 0 ? '+' : '';

  document.getElementById('headerSub').textContent =
    'Aggiornato: ' + (appState.last_update || appState.last_analysis || '—');

  el.innerHTML = `
    <!-- KPI principali -->
    <div class="kpi-row">
      <div class="kpi">
        <div class="val" style="color:var(--blue)">
          ${appState.capital ? '€'+appState.capital.toLocaleString('it-IT') : '—'}
        </div>
        <div class="lbl">Capitale investito</div>
      </div>
      <div class="kpi">
        <div class="val" style="color:${gainColor}">
          ${appState.total_value ? '€'+Number(appState.total_value).toLocaleString('it-IT') : '—'}
        </div>
        <div class="lbl">Valore attuale</div>
      </div>
    </div>
    <div class="kpi-row three">
      <div class="kpi">
        <div class="val" style="color:${gainColor}">${gainSign}€${Math.abs(gain).toLocaleString('it-IT')}</div>
        <div class="lbl">P&L totale</div>
      </div>
      <div class="kpi">
        <div class="val" style="color:${gainColor}">${gainSign}${gainPct}%</div>
        <div class="lbl">Rendimento</div>
      </div>
      <div class="kpi">
        <div class="val" style="color:var(--yellow)">${appState.sharpe || '—'}</div>
        <div class="lbl">Sharpe</div>
      </div>
    </div>

    <!-- Prossima revisione -->
    <div class="card">
      <div class="card-title">Prossima revisione consigliata</div>
      <div style="font-size:20px;font-weight:800;color:var(--purple)">
        ${appState.next_review || '—'}
      </div>
      <div style="font-size:12px;color:var(--muted);margin-top:6px">
        Rendimento atteso: <b style="color:var(--green)">
        ${appState.expected_return ? (appState.expected_return*100).toFixed(1)+'%/anno' : '—'}
        </b>
      </div>
    </div>

    <!-- Titoli in portafoglio -->
    <div class="card">
      <div class="card-title">Titoli in portafoglio (${appState.tickers.length})</div>
      ${renderHoldingsList(appState)}
    </div>

    <!-- Azioni rapide -->
    <button class="btn btn-green btn-small" onclick="showScreen('analisi');startUpdate()">
      🔄 Aggiorna prezzi ora
    </button>
  `;
}

function renderHoldingsList(state){
  const holdings = state.holdings || {};
  const tickers  = state.tickers  || [];
  if(!tickers.length) return '<p style="color:var(--muted);font-size:13px">Nessun titolo</p>';

  return tickers.map(t=>{
    const h    = holdings[t] || {};
    const curr = h.current_price;
    const gain = (curr && h.avg_price && h.quantity)
      ? (curr - h.avg_price) * h.quantity : null;
    const gainPct = (curr && h.avg_price)
      ? ((curr/h.avg_price - 1)*100).toFixed(1) : null;
    const gc = gain === null ? '' : gain >= 0 ? 'badge-green' : 'badge-red';
    const gs = gain === null ? '' : (gain>=0?'+':'')+gain.toFixed(0)+'€ ('+gainPct+'%)';

    return `<div class="ticker-row">
      <div>
        <div class="ticker-name">${t}</div>
        <div class="ticker-sub">
          ${h.quantity ? h.quantity+' az. @ €'+h.avg_price : 'quantità non inserita'}
        </div>
      </div>
      <div>
        ${gc ? `<span class="badge ${gc}">${gs}</span>` :
               `<span class="badge badge-blue">${Math.round((h.weight||0)*100)}%</span>`}
      </div>
    </div>`;
  }).join('');
}

// ══ PORTFOLIO ══════════════════════════════════════════════════
async function loadPortfolio(){
  const r = await fetch('/api/state');
  const s = await r.json();
  const el = document.getElementById('portfolioContent');

  if(!s.tickers || !s.tickers.length){
    el.innerHTML = `<div class="empty">
      <div class="empty-icon">💼</div>
      <h3>Portafoglio vuoto</h3>
      <p>Esegui un'analisi per vedere i titoli consigliati</p>
    </div>`;
    return;
  }

  const holdings = s.holdings || {};
  el.innerHTML = `
    <div class="section-title">Allocazione consigliata</div>
    <div class="card" style="padding:0 18px">
      ${s.tickers.map(t=>{
        const h = holdings[t]||{};
        const w = Math.round((h.weight||0)*100);
        const amt = Math.round((h.amount||0));
        return `<div class="ticker-row">
          <div style="flex:1">
            <div class="ticker-name">${t}</div>
            <div class="ticker-sub">€${amt.toLocaleString('it-IT')}</div>
          </div>
          <div style="display:flex;align-items:center;gap:10px">
            <div style="width:80px;height:6px;background:var(--border);border-radius:3px">
              <div style="width:${w}%;height:100%;background:var(--blue);border-radius:3px"></div>
            </div>
            <span style="font-weight:700;font-size:14px;color:var(--blue);min-width:32px">${w}%</span>
          </div>
        </div>`;
      }).join('')}
    </div>
    <div class="alert alert-info">
      💡 Vai su <b>Impostazioni</b> per inserire le quantità che hai acquistato
      e calcolare il tuo P&L reale.
    </div>
  `;
}

// ══ REPORT ═════════════════════════════════════════════════════
async function loadReport(){
  const el = document.getElementById('reportContent');
  // Prova a caricare il report HTML in un iframe
  const check = await fetch('/api/report');
  if(check.ok){
    el.innerHTML = `
      <div style="padding:12px 0 8px;display:flex;justify-content:space-between;align-items:center">
        <div style="font-size:13px;color:var(--muted)">Report interattivo completo</div>
        <a href="/api/report" target="_blank"
           style="font-size:12px;color:var(--blue);font-weight:600">
          Apri a schermo intero ↗
        </a>
      </div>
      <iframe src="/api/report" class="report-frame"></iframe>
    `;
  } else {
    el.innerHTML = `<div class="empty" style="margin-top:40px">
      <div class="empty-icon">📄</div>
      <h3>Nessun report</h3>
      <p>Esegui un'analisi completa<br>per generare il report</p>
      <button class="btn btn-outline btn-small"
              style="margin-top:20px;width:auto;padding:10px 24px"
              onclick="showScreen('analisi')">Vai ad Analisi</button>
    </div>`;
  }
}

// ══ SETTINGS ═══════════════════════════════════════════════════
async function loadSettings(){
  const r = await fetch('/api/state');
  const s = await r.json();
  const el = document.getElementById('settingsInfo');
  el.innerHTML = `
    <div>📅 Ultima analisi: <b>${s.last_analysis||'Mai'}</b></div>
    <div>🔄 Ultimo aggiornamento: <b>${s.last_update||'—'}</b></div>
    <div>📦 Titoli in portafoglio: <b>${(s.tickers||[]).length}</b></div>
    <div>💰 Capitale: <b>${s.capital ? '€'+s.capital.toLocaleString('it-IT') : '—'}</b></div>
    <div style="margin-top:10px;font-size:11px;color:var(--muted)">
      v7 — Portfolio Optimizer Web App
    </div>
  `;

  const listEl = document.getElementById('holdingsList');
  const tickers = s.tickers || [];
  const holdings = s.holdings || {};
  if(!tickers.length){
    listEl.innerHTML = '<p style="font-size:13px;color:var(--muted)">Nessun titolo nel portafoglio</p>';
    return;
  }
  listEl.innerHTML = tickers.map(t=>{
    const h = holdings[t]||{};
    return `<div style="display:flex;justify-content:space-between;align-items:center;
                         padding:10px 0;border-bottom:1px solid var(--border)">
      <div>
        <div style="font-weight:700">${t}</div>
        <div style="font-size:11px;color:var(--muted)">
          ${h.quantity||0} az. @ €${h.avg_price||0}
        </div>
      </div>
      <button class="btn btn-outline btn-small" style="width:auto;padding:8px 14px"
              onclick="editHolding('${t}',${h.quantity||0},${h.avg_price||0})">
        Modifica
      </button>
    </div>`;
  }).join('');
}

// ══ ANALISI ════════════════════════════════════════════════════
function startAnalysis(){
  const capital   = parseFloat(document.getElementById('inputCapital').value)||10000;
  const maxStocks = parseInt(document.getElementById('inputMaxStocks').value)||12;
  const tickersRaw= document.getElementById('inputTickers').value.trim();
  const tickers   = tickersRaw ? tickersRaw.toUpperCase().split(/\\s+/) : [];

  document.getElementById('btnAnalisi').disabled = true;
  document.getElementById('btnAnalisi').textContent = '⏳ Analisi in corso...';
  showProgress(true);

  fetch('/api/start-analysis',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({capital, max_stocks:maxStocks, tickers})
  })
  .then(r=>r.json())
  .then(d=>pollJob(d.job_id, 'analysis'));
}

function startUpdate(){
  showProgress(true);
  document.getElementById('progressMsg').textContent = '📥 Aggiornamento prezzi in corso...';
  fetch('/api/start-update',{method:'POST'})
    .then(r=>r.json())
    .then(d=>pollJob(d.job_id, 'update'));
}

function showProgress(show){
  document.getElementById('analysisProgress').style.display = show ? 'block' : 'none';
}

function pollJob(jobId, type){
  if(pollingTimer) clearInterval(pollingTimer);
  pollingTimer = setInterval(async ()=>{
    const r   = await fetch('/api/job/'+jobId);
    const job = await r.json();

    document.getElementById('progressFill').style.width = (job.progress||0)+'%';
    document.getElementById('progressPct').textContent  = (job.progress||0)+'%';
    document.getElementById('progressMsg').textContent  = job.message||'';

    if(job.status === 'done'){
      clearInterval(pollingTimer);
      document.getElementById('btnAnalisi').disabled = false;
      document.getElementById('btnAnalisi').textContent = '🚀 Avvia Analisi Completa';
      setTimeout(()=>{ showProgress(false); }, 1500);

      if(type==='analysis' && job.result){
        showScreen('home');
        setTimeout(loadDashboard, 500);
        showToast('✅ Analisi completata! ' + job.result.n_stocks + ' titoli selezionati');
      } else if(type==='update'){
        showToast('✅ Prezzi aggiornati!');
        loadDashboard();
      }
    }
    if(job.status === 'error'){
      clearInterval(pollingTimer);
      showProgress(false);
      document.getElementById('btnAnalisi').disabled = false;
      document.getElementById('btnAnalisi').textContent = '🚀 Avvia Analisi Completa';
      showToast('❌ ' + job.error, 'error');
    }
  }, 1500);
}

// ══ HOLDINGS MODAL ═════════════════════════════════════════════
function openAddHolding(){ openModal('','',''); }
function editHolding(t,q,a){ openModal(t,q,a); }
function openModal(t,q,a){
  document.getElementById('mTicker').value = t;
  document.getElementById('mQty').value    = q||'';
  document.getElementById('mAvg').value    = a||'';
  document.getElementById('modalHolding').classList.add('open');
}
function closeModal(){
  document.getElementById('modalHolding').classList.remove('open');
}
async function saveHolding(){
  const ticker = document.getElementById('mTicker').value.trim().toUpperCase();
  const qty    = parseFloat(document.getElementById('mQty').value)||0;
  const avg    = parseFloat(document.getElementById('mAvg').value)||0;
  if(!ticker){ showToast('Inserisci il ticker','error'); return; }

  await fetch('/api/update-holding',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ticker, quantity:qty, avg_price:avg})
  });
  closeModal();
  loadSettings();
  showToast('✅ Salvato!');
}

// ══ TOAST ══════════════════════════════════════════════════════
function showToast(msg, type='success'){
  const t = document.createElement('div');
  t.textContent = msg;
  t.style.cssText = `
    position:fixed;bottom:calc(80px + env(safe-area-inset-bottom));
    left:50%;transform:translateX(-50%) translateY(20px);
    background:${type==='error'?'#450a0a':'#052e16'};
    color:${type==='error'?'#f87171':'#4ade80'};
    border:1px solid ${type==='error'?'#7f1d1d':'#14532d'};
    padding:12px 20px;border-radius:10px;font-size:13px;font-weight:600;
    z-index:300;white-space:nowrap;transition:all .3s;opacity:0`;
  document.body.appendChild(t);
  requestAnimationFrame(()=>{
    t.style.opacity='1'; t.style.transform='translateX(-50%) translateY(0)';
  });
  setTimeout(()=>{
    t.style.opacity='0';
    setTimeout(()=>t.remove(), 300);
  }, 3000);
}

// ══ INIT ═══════════════════════════════════════════════════════
loadDashboard();
// Aggiorna header ogni 60 sec
setInterval(()=>{ if(currentScreen==='home') loadDashboard(); }, 60000);
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════
# AVVIO
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    print(f"\n{'═'*50}")
    print(f"  📈  Portfolio Optimizer Web App")
    print(f"  Apri sul telefono: http://TUO-IP:{port}")
    print(f"{'═'*50}\n")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
