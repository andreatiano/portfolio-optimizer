"""
Generatore di report: output su console + HTML interattivo a schede.

L'HTML generato è un dashboard completo con:
- 5 tab navigabili (Panoramica, Portafoglio, Backtest, Rischio/Rendimento, Titoli)
- Grafici interattivi (Chart.js caricato da CDN)
- Tabella titoli ordinabile per qualsiasi colonna
- KPI, barre di score, scatter rischio/rendimento
- Zero dipendenze esterne: tutto in un unico file HTML
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger("portfolio_optimizer")


class ReportGenerator:
    """Genera report testuali, grafici PNG e HTML interattivo del portafoglio."""

    SECTOR_COLORS = {
        "Technology":             "#60A5FA",
        "Healthcare":             "#34D399",
        "Financial Services":     "#FBBF24",
        "Consumer Cyclical":      "#F87171",
        "Consumer Defensive":     "#818CF8",
        "Industrials":            "#2DD4BF",
        "Energy":                 "#FB923C",
        "Communication Services": "#38BDF8",
        "Communication":          "#38BDF8",
        "Real Estate":            "#E67E22",
        "Utilities":              "#94A3B8",
        "Basic Materials":        "#4ADE80",
        "Unknown":                "#475569",
    }

    def __init__(self, portfolio, backtest, price_data, ticker_info, scores, capital):
        self.portfolio = portfolio
        self.backtest  = backtest
        self.prices    = price_data
        self.info      = ticker_info
        self.scores    = scores
        self.capital   = capital
        os.makedirs("output", exist_ok=True)

    # ─── CONSOLE ─────────────────────────────────────────────────

    def print_summary(self):
        p  = self.portfolio
        bt = self.backtest

        print("\n" + "═"*65)
        print("  📊  PORTAFOGLIO OTTIMIZZATO")
        print("═"*65)
        print(f"\n  💰 Capitale totale:        {p['capital']:>12,.0f} €")
        print(f"  📈 Rendimento atteso:      {p['expected_return']:>11.1%} /anno")
        print(f"  📉 Volatilità attesa:      {p['expected_volatility']:>11.1%} /anno")
        print(f"  ⚖️  Sharpe Ratio:           {p['sharpe_ratio']:>12.2f}")
        print(f"  🗓  Revisione consigliata:  {p['horizon_label']:>16}")

        print(f"\n  📋 Backtest ({bt['years']}a: {bt['start_date']} → {bt['end_date']})")
        print(f"  ├─ Rendimento totale:    {bt['total_return']:>10.1%}")
        print(f"  ├─ Rendimento annuo:     {bt['annual_return']:>10.1%}")
        print(f"  ├─ Volatilità annua:     {bt['annual_volatility']:>10.1%}")
        print(f"  ├─ Sharpe storico:       {bt['sharpe_ratio']:>10.2f}")
        print(f"  ├─ Max Drawdown:         {bt['max_drawdown']:>10.1%}")
        print(f"  └─ Benchmark (EW):       {bt['benchmark_total']:>10.1%}")

        print(f"\n  📑 TITOLI ({len(p['tickers'])})")
        print("  " + "─"*63)
        print(f"  {'Ticker':<8} {'Nome':<26} {'Peso':>6} {'Importo':>9} {'Score':>6}")
        print("  " + "─"*63)
        for item in p["allocation"]:
            t    = item["ticker"]
            name = self.info.get(t, {}).get("name", t)[:24]
            row  = self.scores[self.scores["ticker"] == t]
            sc   = float(row["composite_score"].iloc[0]) if not row.empty else 0
            print(f"  {t:<8} {name:<26} {item['weight']:>5.1%} {item['amount_eur']:>8,.0f}€  {sc:>5.1f}")

        print(f"\n  🏭 SETTORI")
        print("  " + "─"*40)
        for sec, w in sorted(p["sector_allocation"].items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 30)
            print(f"  {sec:<28} {w:>5.1%}  {bar}")

        print(f"\n  💡 MOTIVAZIONI")
        print("  " + "─"*63)
        for t in p["tickers"][:10]:
            row = self.scores[self.scores["ticker"] == t]
            rat = str(row["rationale"].iloc[0]) if not row.empty else ""
            if rat:
                print(f"  {t:<8} {rat[:56]}")
        print("\n" + "═"*65)

    # ─── GRAFICI PNG ─────────────────────────────────────────────

    def generate_plots(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("   ⚠️  matplotlib non disponibile, grafici PNG saltati")
            return

        fig = plt.figure(figsize=(18, 14))
        fig.suptitle("Portfolio Optimizer — Analisi Completa", fontsize=15, fontweight="bold")
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        self._mpl_backtest(fig.add_subplot(gs[0, :2]))
        self._mpl_pie(fig.add_subplot(gs[0, 2]))
        self._mpl_sector(fig.add_subplot(gs[1, 0]))
        self._mpl_scatter(fig.add_subplot(gs[1, 1:]))
        self._mpl_drawdown(fig.add_subplot(gs[2, :2]))
        self._mpl_scores(fig.add_subplot(gs[2, 2]))

        plt.savefig("output/portfolio_charts.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _mpl_backtest(self, ax):
        import matplotlib.pyplot as plt
        bt    = self.backtest
        cum   = bt["portfolio_cumulative"]  * 100 - 100
        bench = bt["benchmark_cumulative"] * 100 - 100
        ax.plot(cum.index, cum.values,   color="#60A5FA", lw=2,   label="Portafoglio")
        ax.plot(bench.index, bench.values, color="#FBBF24", lw=1.5, ls="--", label="Benchmark EW")
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.fill_between(cum.index, cum.values, 0, alpha=0.1, color="#60A5FA")
        ax.set_title(f"Performance Storica ({bt['years']} anni)")
        ax.set_ylabel("Rendimento (%)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    def _mpl_pie(self, ax):
        import matplotlib.pyplot as plt
        items  = self.portfolio["allocation"][:12]
        labels = [i["ticker"] for i in items]
        sizes  = [i["weight"]  for i in items]
        colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
        ax.pie(sizes, labels=labels, autopct="%1.0f%%", colors=colors,
               startangle=90, pctdistance=0.75, textprops={"fontsize": 7})
        ax.set_title("Allocazione Titoli")

    def _mpl_sector(self, ax):
        sa      = self.portfolio["sector_allocation"]
        sectors = list(sa.keys())
        weights = [sa[s] * 100 for s in sectors]
        colors  = [self.SECTOR_COLORS.get(s, "#BDC3C7") for s in sectors]
        bars    = ax.barh(sectors, weights, color=colors)
        for bar, w in zip(bars, weights):
            ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                    f"{w:.1f}%", va="center", fontsize=7)
        ax.set_xlim(0, max(weights)*1.25)
        ax.set_title("Settori"); ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.3, axis="x")

    def _mpl_scatter(self, ax):
        sc = self.scores[self.scores["ticker"].isin(self.portfolio["tickers"])].copy()
        ax.scatter(
            sc["vol_annual"].fillna(0.3) * 100,
            sc["cagr_5y"].fillna(0.08)  * 100,
            c=[self.SECTOR_COLORS.get(s, "#888") for s in sc["sector"]],
            s=[self.portfolio["weights"].get(t, 0.05)*3000 for t in sc["ticker"]],
            alpha=0.75, edgecolors="white", lw=0.5
        )
        for _, row in sc.iterrows():
            ax.annotate(row["ticker"],
                        (row.get("vol_annual", 0.3)*100, row.get("cagr_5y", 0.08)*100),
                        xytext=(5, 3), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Volatilità (%)"); ax.set_ylabel("CAGR 5A (%)")
        ax.set_title("Rischio / Rendimento"); ax.grid(True, alpha=0.3)

    def _mpl_drawdown(self, ax):
        bt  = self.backtest
        cum = bt["portfolio_cumulative"]
        dd  = (cum / cum.expanding().max() - 1) * 100
        ax.fill_between(dd.index, dd.values, 0, color="#F87171", alpha=0.6)
        ax.plot(dd.index, dd.values, color="#DC2626", lw=0.8)
        ax.set_title(f"Drawdown (Max: {bt['max_drawdown']:.1%})")
        ax.set_ylabel("Drawdown (%)"); ax.grid(True, alpha=0.3)

    def _mpl_scores(self, ax):
        sc = self.scores[self.scores["ticker"].isin(self.portfolio["tickers"])].head(12)
        sc = sc.sort_values("composite_score")
        colors = ["#34D399" if s >= 75 else "#FBBF24" if s >= 60 else "#F87171"
                  for s in sc["composite_score"]]
        ax.barh(sc["ticker"], sc["composite_score"], color=colors)
        ax.axvline(50, color="gray", lw=0.8, ls="--")
        ax.set_title("Score Composito"); ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, axis="x")

    def _compute_forecast(self, bt: dict, p: dict) -> dict:
        """
        Simulazione Monte Carlo per previsione portafoglio a 1/3/5 anni.
        Usa rendimento atteso e volatilità del portafoglio ottimizzato.
        """
        try:
            mu_daily  = p["expected_return"]  / 252
            vol_daily = p["expected_volatility"] / (252 ** 0.5)
            capital   = self.capital

            horizons = {"1y": 252, "3y": 756, "5y": 1260}
            n_sims   = 500
            results  = {}

            for label, days in horizons.items():
                rng      = np.random.default_rng(42)
                daily_r  = rng.normal(mu_daily, vol_daily, (n_sims, days))
                final_v  = capital * np.prod(1 + daily_r, axis=1)
                results[label] = {
                    "p10":    round(float(np.percentile(final_v, 10)),  0),
                    "p25":    round(float(np.percentile(final_v, 25)),  0),
                    "p50":    round(float(np.percentile(final_v, 50)),  0),
                    "p75":    round(float(np.percentile(final_v, 75)),  0),
                    "p90":    round(float(np.percentile(final_v, 90)),  0),
                    "mean":   round(float(np.mean(final_v)),            0),
                    "prob_positive": round(float(np.mean(final_v > capital)) * 100, 1),
                }

            return results

        except Exception as e:
            logger.warning(f"Forecast Monte Carlo fallito: {e}")
            return {}

    # ─── HTML ────────────────────────────────────────────────────

    def save_report(self):
        """Genera output/portfolio_report.html — dashboard interattivo completo."""
        payload = self._build_json_payload()
        html    = _HTML_TEMPLATE.replace("__PAYLOAD__", payload)
        path    = "output/portfolio_report.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Report HTML salvato: {path}")

    def _build_json_payload(self) -> str:
        p  = self.portfolio
        bt = self.backtest

        # ── Scarica S&P 500 (SPY) per stesso periodo backtest ────
        spy_ts_map = {}
        try:
            import yfinance as yf
            cum_p_ref  = bt["portfolio_cumulative"]
            start_date = str(cum_p_ref.index[0])[:10]
            end_date   = str(cum_p_ref.index[-1])[:10]
            spy_raw = yf.download("SPY", start=start_date, end=end_date,
                                  auto_adjust=True, progress=False)
            if not spy_raw.empty:
                spy_close = spy_raw["Close"].squeeze()
                spy_rets  = spy_close.pct_change().fillna(0)
                spy_cum   = (1 + spy_rets).cumprod()
                spy_cum   = spy_cum / spy_cum.iloc[0]
                spy_ts_map = {str(d)[:10]: round(float(v)*100-100, 2)
                              for d, v in spy_cum.items()}
        except Exception as e:
            logger.warning(f"Download SPY fallito: {e}")

        # ── Genera previsione Monte Carlo ─────────────────────────
        forecast_data = self._compute_forecast(bt, p)

        # Timeseries (ogni 5 giorni)
        cum_p = bt["portfolio_cumulative"]
        cum_b = bt["benchmark_cumulative"]
        ts = []
        for i, (d, v) in enumerate(cum_p.items()):
            if i % 5 == 0:
                d_str = str(d)[:10]
                spy_val = spy_ts_map.get(d_str)
                if spy_val is None:
                    from datetime import timedelta as _td
                    for off in [1,-1,2,-2,3,-3,4,-4,5,-5]:
                        alt = str((d + _td(days=off)))[:10]
                        if alt in spy_ts_map:
                            spy_val = spy_ts_map[alt]; break
                ts.append({
                    "date":      d_str[:7],
                    "portfolio": round(float(v)*100-100, 2),
                    "benchmark": round(float(cum_b.iloc[i])*100-100, 2),
                    "sp500":     spy_val,
                })

        # Drawdown
        peak = cum_p.expanding().max()
        dd_s = (cum_p / peak - 1)
        dd   = [{"date": str(d)[:7], "dd": round(float(v)*100, 2)}
                for i, (d, v) in enumerate(dd_s.items()) if i % 5 == 0]

        # Stocks
        def safe(v):
            try:
                return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
            except Exception:
                return None

        stocks_out = []
        for ticker in p["tickers"]:
            row  = self.scores[self.scores["ticker"] == ticker]
            info = self.info.get(ticker, {})
            if row.empty:
                continue
            r = row.iloc[0]
            stocks_out.append({
                "ticker":            ticker,
                "name":              info.get("name", ticker),
                "sector":            info.get("sector", "Unknown"),
                "country":           info.get("country", "US"),
                "composite_score":   round(float(r["composite_score"]), 1),
                "technical_score":   round(float(r["technical_score"]),  1),
                "fundamental_score": round(float(r["fundamental_score"]),1),
                "cagr_5y":           safe(r.get("cagr_5y")),
                "vol_annual":        safe(r.get("vol_annual")),
                "max_drawdown":      safe(r.get("max_drawdown")),
                "sharpe_ratio":      round(float(r.get("sharpe_ratio", 0)), 2),
                "pe_ratio":          safe(r.get("pe_ratio")),
                "roe":               safe(r.get("roe")),
                "revenue_growth":    safe(r.get("revenue_growth")),
                "earnings_growth":   safe(r.get("earnings_growth")),
                "fcf_yield":         safe(r.get("fcf_yield")),
                "rationale":         str(r.get("rationale", "")),
                "weight":            round(p["weights"].get(ticker, 0), 4),
                "amount":            round(p["weights"].get(ticker, 0) * self.capital, 0),
            })

        # News per ticker (se disponibili)
        news_out = {}
        for ticker in p["tickers"]:
            row = self.scores[self.scores["ticker"] == ticker]
            if row.empty:
                continue
            r = row.iloc[0]
            # news_raw_score va da -100 a +100
            raw_ns   = r.get("news_raw_score", 0) or 0
            ns_norm  = r.get("news_score", 50) or 50  # 0..100 normalizzato
            kp = r.get("news_key_points") or []
            rf = r.get("news_risk_flags") or []
            op = r.get("news_opportunity") or []
            kp = list(kp) if hasattr(kp, '__iter__') and not isinstance(kp, str) else []
            rf = list(rf) if hasattr(rf, '__iter__') and not isinstance(rf, str) else []
            op = list(op) if hasattr(op, '__iter__') and not isinstance(op, str) else []
            news_out[ticker] = {
                "signal":           str(r.get("news_signal", "NEUTRO")),
                "sentiment":        str(r.get("news_sentiment", "Neutro")),
                "news_score":       round(float(ns_norm), 1),
                "news_raw_score":   int(raw_ns),         # -100..+100
                "article_count":    int(r.get("news_articles", 0) or 0),
                "summary":          str(r.get("news_summary", "") or ""),
                "key_points":       kp[:3],
                "risk_flags":       rf[:3],
                "opportunity":      op[:3],
                "timing":           str(r.get("news_timing", "") or ""),
                "favorable":        bool(r.get("news_favorable")) if r.get("news_favorable") is not None else None,
            }

        payload = {
            "generated": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "capital":   self.capital,
            "news":      news_out,
            "forecast":  forecast_data,
            "portfolio": {
                "expected_return":     round(p["expected_return"],      4),
                "expected_volatility": round(p["expected_volatility"],  4),
                "sharpe_ratio":        round(p["sharpe_ratio"],         4),
                "horizon_label":       p["horizon_label"],
                "next_review":         p["next_review"],
                "sector_allocation":   {k: round(v,4) for k,v in p["sector_allocation"].items()},
                "allocation": [
                    {"ticker": a["ticker"],
                     "weight": round(a["weight"], 4),
                     "amount": round(a["amount_eur"], 0)}
                    for a in p["allocation"]
                ],
            },
            "backtest": {
                "total_return":      round(bt["total_return"],      4),
                "annual_return":     round(bt["annual_return"],     4),
                "annual_volatility": round(bt["annual_volatility"], 4),
                "sharpe_ratio":      round(bt["sharpe_ratio"],      4),
                "max_drawdown":      round(bt["max_drawdown"],      4),
                "benchmark_total":   round(bt["benchmark_total"],   4),
                "calmar_ratio":      round(bt.get("calmar_ratio",0),4),
                "start_date":        bt["start_date"],
                "end_date":          bt["end_date"],
                "years":             bt["years"],
                "timeseries":        ts,
                "drawdown_series":   dd,
            },
            "stocks": stocks_out,
        }
        return json.dumps(payload, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE — viene popolato con __PAYLOAD__ dal metodo save_report()
# ═════════════════════════════════════════════════════════════════════════════

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Optimizer — Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#060d1a;--bg1:#0a1628;--bg2:#0f172a;--border:#1e293b;
    --text:#e2e8f0;--muted:#64748b;--dim:#475569;
    --blue:#60A5FA;--green:#34D399;--yellow:#FBBF24;
    --purple:#818CF8;--red:#F87171;--teal:#2DD4BF;
  }
  html,body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px;line-height:1.5}
  .wrap{max-width:1300px;margin:0 auto;padding:28px 20px}
  /* header */
  .header{display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;margin-bottom:28px}
  .header h1{font-size:26px;font-weight:900;letter-spacing:-.5px;background:linear-gradient(90deg,var(--blue),var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .header .sub{font-size:12px;color:var(--muted);margin-top:4px}
  .badge{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:6px 14px;font-size:12px;color:var(--green)}
  .badge.blue{color:var(--blue)}
  /* tabs */
  .tabs{display:flex;gap:4px;background:var(--bg2);padding:4px;border-radius:10px;border:1px solid var(--border);width:fit-content;margin-bottom:24px;flex-wrap:wrap}
  .tab{padding:8px 18px;border-radius:7px;border:none;cursor:pointer;font-size:13px;font-weight:600;color:var(--muted);background:transparent;transition:all .2s}
  .tab.active{background:#1e40af;color:#bfdbfe}
  .tab:hover:not(.active){color:var(--text)}
  /* panels */
  .panel{display:none}.panel.active{display:block}
  /* kpi */
  .kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin-bottom:20px}
  .kpi{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:20px 18px}
  .kpi .lbl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:8px}
  .kpi .val{font-size:26px;font-weight:800;font-family:'SF Mono','Fira Code',monospace;letter-spacing:-1px}
  .kpi .sub{font-size:11px;color:var(--dim);margin-top:5px}
  /* card */
  .card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:24px;margin-bottom:18px}
  .card-title{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:16px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
  /* bt stat */
  .bt-stat{background:var(--bg1);border-radius:8px;padding:14px 16px;text-align:center}
  .bt-stat .lbl{font-size:11px;color:var(--muted);margin-bottom:6px}
  .bt-stat .val{font-size:22px;font-weight:800;font-family:monospace}
  /* table */
  .tbl-wrap{overflow-x:auto}
  table{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}
  th{padding:10px 12px;text-align:left;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;border-bottom:1px solid var(--border);cursor:pointer;user-select:none;background:#060d1a}
  th:hover{color:var(--blue)}
  td{padding:11px 12px;border-bottom:1px solid var(--border);vertical-align:middle}
  tr:hover td{background:#111d33}
  .tbadge{display:inline-block;padding:3px 9px;border-radius:6px;font-weight:700;font-size:12px}
  /* score bars */
  .sbar-wrap{display:flex;align-items:center;gap:7px;min-width:90px}
  .sbar-bg{flex:1;height:5px;background:var(--border);border-radius:99px;overflow:hidden}
  .sbar-fill{height:100%;border-radius:99px}
  /* sector bars */
  .sec-row{margin-bottom:10px}
  .sec-hdr{display:flex;justify-content:space-between;margin-bottom:4px;font-size:12px}
  .sec-bar-bg{height:5px;background:var(--border);border-radius:99px;overflow:hidden}
  .sec-bar{height:100%;border-radius:99px}
  /* alloc rows */
  .alloc-row{display:grid;grid-template-columns:90px 1fr 70px 80px 90px;align-items:center;gap:12px;padding:12px 16px;background:var(--bg1);border-radius:8px;border:1px solid var(--border);margin-bottom:8px}
  .alloc-row:hover{border-color:#334155}
  /* disclaimer */
  .disclaimer{margin-top:32px;padding:12px 16px;background:var(--bg1);border:1px solid var(--border);border-radius:8px;font-size:11px;color:#334155;text-align:center}
  .disclaimer strong{color:var(--dim)}.disclaimer code{color:var(--blue)}
  @media(max-width:700px){.grid2,.grid3{grid-template-columns:1fr}.alloc-row{grid-template-columns:80px 1fr 60px}.hide-sm{display:none}}
</style>
</head>
<body>
<div class="wrap">

<div class="header">
  <div>
    <h1>📊 Portfolio Optimizer</h1>
    <div class="sub" id="hdrSub"></div>
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap" id="hdrBadges"></div>
</div>

<div class="tabs">
  <button class="tab active" onclick="setTab(0)">Panoramica</button>
  <button class="tab" onclick="setTab(1)">Portafoglio</button>
  <button class="tab" onclick="setTab(2)">Backtest</button>
  <button class="tab" onclick="setTab(3)">Rischio / Rendimento</button>
  <button class="tab" onclick="setTab(4)">Titoli</button>
  <button class="tab" onclick="setTab(5)">📰 Notizie</button>
  <button class="tab" onclick="setTab(6)">🔮 Previsione</button>
</div>

<!-- ── TAB 0 ── -->
<div class="panel active" id="tab0">
  <div class="kpi-grid" id="kpiGrid0"></div>
  <div class="card">
    <div class="card-title" id="btTitle0"></div>
    <div class="grid3" id="btStats0" style="margin-bottom:20px"></div>
    <div style="position:relative;height:210px"><canvas id="cOverview"></canvas></div>
  </div>
  <div class="grid2">
    <div class="card">
      <div class="card-title">Distribuzione settoriale</div>
      <div id="secBars0"></div>
    </div>
    <div class="card">
      <div class="card-title">Distribuzione geografica globale</div>
      <div style="position:relative;height:210px"><canvas id="cRegion"></canvas></div>
    </div>
  </div>
</div>

<!-- ── TAB 1 ── -->
<div class="panel" id="tab1">
  <div class="grid2">
    <div class="card">
      <div class="card-title">Allocazione per titolo</div>
      <div style="position:relative;height:290px"><canvas id="cAlloc"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Distribuzione settoriale</div>
      <div style="position:relative;height:290px"><canvas id="cSector"></canvas></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Piano di investimento dettagliato</div>
    <div id="allocRows"></div>
    <div style="margin-top:12px;padding:10px 16px;background:#060d1a;border-radius:8px;font-size:12px;color:var(--muted)" id="costNote"></div>
  </div>
</div>

<!-- ── TAB 2 ── -->
<div class="panel" id="tab2">
  <div class="kpi-grid" id="kpiGrid2"></div>
  <div class="card">
    <div class="card-title">Performance storica: Portafoglio · Benchmark EW · S&P 500</div>
    <div style="position:relative;height:300px"><canvas id="cBacktest"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title" id="ddTitle"></div>
    <div style="font-size:11px;color:#334155;margin-bottom:16px" id="ddSub"></div>
    <div style="position:relative;height:150px"><canvas id="cDD"></canvas></div>
  </div>
</div>

<!-- ── TAB 3 ── -->
<div class="panel" id="tab3">
  <div class="card">
    <div class="card-title">Mappa Rischio / Rendimento</div>
    <div style="font-size:11px;color:var(--dim);margin-bottom:16px">
      Asse X = Volatilità annua &nbsp;·&nbsp; Asse Y = CAGR 5 anni &nbsp;·&nbsp; Dimensione = Peso in portafoglio
    </div>
    <div style="position:relative;height:340px"><canvas id="cScatter"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Score composito, tecnico e fondamentale per titolo</div>
    <div id="scoreBars"></div>
  </div>
</div>

<!-- ── TAB 4 ── -->
<div class="panel" id="tab4">
  <div class="card" style="padding:0">
    <div class="tbl-wrap">
      <table id="stocksTable">
        <thead><tr>
          <th onclick="srt('ticker')">Ticker</th>
          <th>Nome</th>
          <th onclick="srt('composite_score')">Score ↕</th>
          <th onclick="srt('weight')">Peso ↕</th>
          <th onclick="srt('cagr_5y')">CAGR 5A ↕</th>
          <th onclick="srt('vol_annual')">Vol ↕</th>
          <th onclick="srt('max_drawdown')">Max DD ↕</th>
          <th onclick="srt('sharpe_ratio')">Sharpe ↕</th>
          <th onclick="srt('pe_ratio')">P/E ↕</th>
          <th onclick="srt('roe')">ROE ↕</th>
          <th onclick="srt('revenue_growth')">Rev Growth ↕</th>
          <th onclick="srt('fcf_yield')">FCF Yield ↕</th>
          <th>Motivazione</th>
        </tr></thead>
        <tbody id="stocksTbody"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="panel" id="tab5">
  <div class="card">
    <div class="card-title">Sentiment Notizie — Titoli in Portafoglio</div>
    <div id="newsSummaryBar" style="display:flex;gap:16px;margin-bottom:20px;flex-wrap:wrap"></div>
  </div>
  <div id="newsGrid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px"></div>
</div>

<!-- ── TAB 6 Previsione ── -->
<div class="panel" id="tab6">
  <div class="kpi-grid" id="kpiGridFc"></div>
  <div class="card">
    <div class="card-title">🔮 Proiezione Monte Carlo — Rendimento Futuro Atteso</div>
    <div style="font-size:12px;color:var(--dim);margin-bottom:16px" id="fcSubtitle"></div>
    <div style="position:relative;height:340px"><canvas id="cForecast"></canvas></div>
  </div>
  <div class="grid2">
    <div class="card">
      <div class="card-title">Distribuzione rendimento finale</div>
      <div style="position:relative;height:220px"><canvas id="cFcDist"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Interpretazione scenari</div>
      <div id="fcScenarios" style="display:flex;flex-direction:column;gap:10px;margin-top:4px"></div>
    </div>
  </div>
  <div class="card" style="background:#0a1628;border-color:#1e3a5f">
    <div style="font-size:11px;color:#334155;line-height:1.7">
      ⚠️ <strong style="color:#475569">Disclaimer previsione:</strong> Le proiezioni sono generate con simulazione Monte Carlo
      basata su rendimento e volatilità storici del portafoglio. Non costituiscono garanzia di rendimento futuro.
      I mercati finanziari sono soggetti a rischi non modellabili statisticamente.
      Risultati passati non sono indicativi di quelli futuri.
    </div>
  </div>
</div>

<div class="disclaimer">
  ⚠️ Questo report è generato automaticamente a scopo informativo e non costituisce consulenza finanziaria.
  Per dati reali: <code>python main.py</code> &nbsp;·&nbsp; Generato il <span id="genDate"></span>
</div>

</div>

<script>
const D = __PAYLOAD__;

const SC = {
  "Technology":"#60A5FA","Healthcare":"#34D399","Financial Services":"#FBBF24",
  "Consumer Cyclical":"#F87171","Consumer Defensive":"#818CF8","Industrials":"#2DD4BF",
  "Energy":"#FB923C","Communication Services":"#38BDF8","Communication":"#38BDF8",
  "Real Estate":"#E67E22","Utilities":"#94A3B8","Basic Materials":"#4ADE80","Unknown":"#475569"
};

// ── utils ──────────────────────────────────────────────────────────────────
const pct  = (v,d=1) => v==null?"—":(v*100).toFixed(d)+"%";
const pct2 = (v,d=1) => v==null?"—":((v>=0?"+":"")+(v*100).toFixed(d)+"%");
const num  = (v,d=2) => v==null?"—":v.toFixed(d);
const eur  = v => v==null?"—":"€"+v.toLocaleString("it-IT");
const sc   = s => SC[s]||"#64748b";
const h2r  = (h,a) => {
  const r=parseInt(h.slice(1,3),16),g=parseInt(h.slice(3,5),16),b=parseInt(h.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
};
const chartBase = {
  plugins:{
    legend:{labels:{color:"#94a3b8",font:{size:11}}},
    tooltip:{backgroundColor:"#0f172a",borderColor:"#334155",borderWidth:1,
             titleColor:"#94a3b8",bodyColor:"#e2e8f0",padding:10}
  },
  scales:{
    x:{ticks:{color:"#475569",font:{size:10}},grid:{color:"#1e293b"}},
    y:{ticks:{color:"#475569",font:{size:10}},grid:{color:"#1e293b"}}
  }
};

// ── tabs ───────────────────────────────────────────────────────────────────
let activeTab=0, built={};
function setTab(i){
  document.querySelectorAll(".panel").forEach((p,j)=>p.classList.toggle("active",j===i));
  document.querySelectorAll(".tab").forEach((t,j)=>t.classList.toggle("active",j===i));
  activeTab=i;
  if(!built[i]){built[i]=true;buildTab(i);}
}
function buildTab(i){[b0,b1,b2,b3,b4,b5,b6][i]();}

// ── TAB 0: Panoramica ─────────────────────────────────────────────────────
function b0(){
  const p=D.portfolio,bt=D.backtest,n=D.stocks.length;
  document.getElementById("hdrSub").textContent=
    `Strategia: Crescita Stabile · ${n} titoli · Capitale ${eur(D.capital)} · Generato ${D.generated}`;
  document.getElementById("hdrBadges").innerHTML=
    `<span class="badge">${n} titoli selezionati</span>
     <span class="badge blue">Revisione: ${p.next_review}</span>`;
  document.getElementById("genDate").textContent=D.generated;

  const kpis=[
    {l:"Capitale",v:eur(D.capital),s:"investito",c:"var(--blue)"},
    {l:"Rendimento atteso",v:pct(p.expected_return),s:"annualizzato",c:"var(--green)"},
    {l:"Volatilità attesa",v:pct(p.expected_volatility),s:"annualizzata",c:"var(--yellow)"},
    {l:"Sharpe Ratio",v:num(p.sharpe_ratio),s:"risk-adjusted",c:"var(--purple)"},
    {l:"N° Titoli",v:n,s:"diversificati",c:"var(--teal)"},
    {l:"Prossima revisione",v:p.next_review,s:p.horizon_label,c:"var(--red)"},
  ];
  document.getElementById("kpiGrid0").innerHTML=kpis.map(k=>
    `<div class="kpi"><div class="lbl">${k.l}</div>
     <div class="val" style="color:${k.c}">${k.v}</div>
     <div class="sub">${k.s}</div></div>`).join("");

  document.getElementById("btTitle0").textContent=
    `📈 Backtest ${bt.years} anni (${bt.start_date} → ${bt.end_date})`;
  const alpha=bt.total_return-bt.benchmark_total;
  const bts=[
    {l:"Rendimento totale",v:pct2(bt.total_return),c:"var(--green)"},
    {l:"Alpha vs benchmark",v:pct2(alpha),c:alpha>=0?"var(--green)":"var(--red)"},
    {l:"Max Drawdown",v:pct(bt.max_drawdown),c:"var(--red)"},
  ];
  document.getElementById("btStats0").innerHTML=bts.map(s=>
    `<div class="bt-stat"><div class="lbl">${s.l}</div>
     <div class="val" style="color:${s.c}">${s.v}</div></div>`).join("");

  const ts=bt.timeseries;
  const ovDatasets=[
    {label:"Portafoglio",data:ts.map(d=>d.portfolio),borderColor:"#60A5FA",
     backgroundColor:h2r("#60A5FA",0.12),fill:true,tension:0.3,pointRadius:0,borderWidth:2.5},
    {label:"Benchmark EW",data:ts.map(d=>d.benchmark),borderColor:"#FBBF24",
     backgroundColor:"transparent",fill:false,tension:0.3,pointRadius:0,
     borderWidth:1.5,borderDash:[6,3]},
  ];
  if(ts.some(d=>d.sp500!=null)){
    ovDatasets.push({label:"S&P 500",data:ts.map(d=>d.sp500),borderColor:"#34D399",
     backgroundColor:"transparent",fill:false,tension:0.3,pointRadius:0,borderWidth:1.5,borderDash:[3,2]});
  }
  new Chart(document.getElementById("cOverview"),{
    type:"line",
    data:{labels:ts.map(d=>d.date),datasets:ovDatasets},
    options:{...chartBase,responsive:true,maintainAspectRatio:false,
      scales:{...chartBase.scales,y:{...chartBase.scales.y,ticks:{...chartBase.scales.y.ticks,callback:v=>v+"%"}}}}
  });

  const sa=p.sector_allocation;
  document.getElementById("secBars0").innerHTML=
    Object.entries(sa).sort((a,b)=>b[1]-a[1]).map(([s,w])=>
      `<div class="sec-row">
       <div class="sec-hdr">
         <span style="color:#94a3b8">${s}</span>
         <span style="color:${sc(s)};font-weight:700">${pct(w)}</span>
       </div>
       <div class="sec-bar-bg"><div class="sec-bar" style="width:${w*300}%;background:${sc(s)}"></div></div>
       </div>`).join("");

  // Raggruppa per macro-area geografica
  const GEO = {
    "USA":          ["US","United States"],
    "Europa":       ["GB","DE","FR","CH","NL","IT","ES","SE","DK","IE","BE","NO","FI","AT","PT"],
    "Giappone":     ["JP","Japan"],
    "Asia-Pac.":    ["AU","SG","HK","CN","TW","KR","IN","South Korea","Australia","Singapore","China","India"],
    "Canada":       ["CA","Canada"],
    "Altri EM":     [], // tutto il resto
  };
  const GEO_COLORS = {
    "USA":"#60A5FA","Europa":"#818CF8","Giappone":"#34D399",
    "Asia-Pac.":"#FBBF24","Canada":"#2DD4BF","Altri EM":"#F87171"
  };
  function getRegion(country){
    for(const [region, codes] of Object.entries(GEO)){
      if(region==="Altri EM") continue;
      if(codes.some(c=>country===c||country.includes(c)||c.includes(country))) return region;
    }
    // fallback per nomi country completi
    if(["United States","US"].includes(country)) return "USA";
    return "Altri EM";
  }
  const regW={};
  D.stocks.forEach(s=>{
    const r=getRegion(s.country||"");
    regW[r]=(regW[r]||0)+s.weight;
  });
  const regEntries=Object.entries(regW).filter(([,w])=>w>0.001).sort((a,b)=>b[1]-a[1]);
  new Chart(document.getElementById("cRegion"),{
    type:"doughnut",
    data:{
      labels:regEntries.map(([r])=>r),
      datasets:[{
        data:regEntries.map(([,w])=>+(w*100).toFixed(1)),
        backgroundColor:regEntries.map(([r])=>GEO_COLORS[r]||"#94a3b8"),
        borderColor:"#0f172a",borderWidth:3
      }]
    },
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{...chartBase.plugins,legend:{display:true,position:"bottom",
        labels:{color:"#94a3b8",font:{size:10},boxWidth:10,
          generateLabels:chart=>chart.data.labels.map((l,i)=>({
            text:l+": "+chart.data.datasets[0].data[i].toFixed(1)+"%",
            fillStyle:chart.data.datasets[0].backgroundColor[i],strokeStyle:"transparent"
          }))
        }}}}
  });
}

// ── TAB 1: Portafoglio ────────────────────────────────────────────────────
function b1(){
  const p=D.portfolio;
  new Chart(document.getElementById("cAlloc"),{
    type:"bar",
    data:{labels:p.allocation.map(a=>a.ticker),
      datasets:[{label:"Peso %",data:p.allocation.map(a=>+(a.weight*100).toFixed(1)),
        backgroundColor:p.allocation.map((_,i)=>`hsl(${210+i*15},70%,60%)`),
        borderRadius:4,borderSkipped:false}]},
    options:{indexAxis:"y",responsive:true,maintainAspectRatio:false,...chartBase,
      plugins:{...chartBase.plugins,legend:{display:false}},
      scales:{...chartBase.scales,x:{...chartBase.scales.x,ticks:{...chartBase.scales.x.ticks,callback:v=>v+"%"}}}}
  });

  const sa=p.sector_allocation,sk=Object.keys(sa);
  new Chart(document.getElementById("cSector"),{
    type:"pie",
    data:{labels:sk,datasets:[{data:sk.map(k=>+(sa[k]*100).toFixed(1)),
      backgroundColor:sk.map(k=>sc(k)),borderColor:"#0f172a",borderWidth:3}]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{...chartBase.plugins,
        legend:{display:true,position:"right",labels:{color:"#94a3b8",font:{size:10},boxWidth:12}},
        tooltip:{...chartBase.plugins.tooltip,callbacks:{label:ctx=>" "+ctx.label+": "+ctx.parsed.toFixed(1)+"%"}}
      }}
  });

  document.getElementById("allocRows").innerHTML=p.allocation.map(item=>{
    const s=D.stocks.find(x=>x.ticker===item.ticker)||{};
    const c=sc(s.sector||"Unknown");
    const cp=item.amount>0?(1/item.amount*100).toFixed(3):"—";
    const cc=parseFloat(cp)<0.1?"var(--green)":"var(--yellow)";
    return `<div class="alloc-row">
      <div><span class="tbadge" style="background:${c}22;color:${c}">${item.ticker}</span></div>
      <div style="font-size:12px;color:#94a3b8">${s.name||item.ticker}</div>
      <div style="text-align:right;font-weight:700">${pct(item.weight)}</div>
      <div style="text-align:right;color:var(--blue);font-weight:700">${eur(item.amount)}</div>
      <div class="hide-sm" style="text-align:right;font-size:11px;color:${cc}">costo: ${cp}%</div>
    </div>`;
  }).join("");

  document.getElementById("costNote").innerHTML=
    `💡 Costo transazioni: <span style="color:var(--green);font-weight:700">€${p.allocation.length}</span>
     &nbsp;·&nbsp; Incidenza: <span style="color:var(--green);font-weight:700">${(p.allocation.length/D.capital*100).toFixed(3)}%</span>`;
}

// ── TAB 2: Backtest ───────────────────────────────────────────────────────
function b2(){
  const bt=D.backtest;
  const alpha=bt.total_return-bt.benchmark_total;
  const items=[
    {l:"Rendimento totale",v:pct2(bt.total_return),s:`benchmark: ${pct2(bt.benchmark_total)}`,c:"var(--green)"},
    {l:"Rendimento annuo",v:pct2(bt.annual_return),s:"CAGR",c:"var(--blue)"},
    {l:"Sharpe storico",v:num(bt.sharpe_ratio),s:"risk-adjusted",c:"var(--purple)"},
    {l:"Volatilità annua",v:pct(bt.annual_volatility),s:"realizzata",c:"var(--yellow)"},
    {l:"Max Drawdown",v:pct(bt.max_drawdown),s:"perdita massima",c:"var(--red)"},
    {l:"Alpha vs benchmark",v:pct2(alpha),s:"outperformance",c:alpha>=0?"var(--green)":"var(--red)"},
  ];
  document.getElementById("kpiGrid2").innerHTML=items.map(k=>
    `<div class="kpi"><div class="lbl">${k.l}</div>
     <div class="val" style="color:${k.c}">${k.v}</div>
     <div class="sub">${k.s}</div></div>`).join("");

  const ts=bt.timeseries;
  const hasSpy = ts.some(d=>d.sp500!=null);
  const btDatasets=[
    {label:"Portafoglio",data:ts.map(d=>d.portfolio),borderColor:"#60A5FA",
     backgroundColor:h2r("#60A5FA",0.12),fill:true,tension:0.3,pointRadius:0,borderWidth:2.5},
    {label:"Benchmark EW",data:ts.map(d=>d.benchmark),borderColor:"#FBBF24",
     backgroundColor:"transparent",fill:false,tension:0.3,pointRadius:0,
     borderWidth:1.5,borderDash:[6,3]},
  ];
  if(hasSpy){
    btDatasets.push({
      label:"S&P 500 (SPY)",data:ts.map(d=>d.sp500),borderColor:"#34D399",
      backgroundColor:"transparent",fill:false,tension:0.3,pointRadius:0,
      borderWidth:1.8,borderDash:[3,2]
    });
  }
  new Chart(document.getElementById("cBacktest"),{
    type:"line",
    data:{labels:ts.map(d=>d.date),datasets:btDatasets},
    options:{responsive:true,maintainAspectRatio:false,...chartBase,
      scales:{...chartBase.scales,
        x:{...chartBase.scales.x,ticks:{...chartBase.scales.x.ticks,maxTicksLimit:12}},
        y:{...chartBase.scales.y,ticks:{...chartBase.scales.y.ticks,callback:v=>v+"%"}}}}
  });

  document.getElementById("ddTitle").innerHTML=
    `Drawdown storico &nbsp;<span style="color:var(--red)">Max: ${pct(bt.max_drawdown)}</span>`;
  document.getElementById("ddSub").textContent=
    "La profondità del drawdown misura la resilienza del portafoglio nei momenti di crisi";

  const dd=bt.drawdown_series;
  new Chart(document.getElementById("cDD"),{
    type:"line",
    data:{labels:dd.map(d=>d.date),
      datasets:[{label:"Drawdown",data:dd.map(d=>d.dd),borderColor:"#F87171",
        backgroundColor:h2r("#F87171",0.35),fill:true,tension:0.3,pointRadius:0,borderWidth:1.5}]},
    options:{responsive:true,maintainAspectRatio:false,...chartBase,
      plugins:{...chartBase.plugins,legend:{display:false}},
      scales:{...chartBase.scales,
        x:{...chartBase.scales.x,ticks:{...chartBase.scales.x.ticks,maxTicksLimit:10}},
        y:{...chartBase.scales.y,ticks:{...chartBase.scales.y.ticks,callback:v=>v+"%"}}}}
  });
}

// ── TAB 3: Rischio/Rendimento ─────────────────────────────────────────────
function b3(){
  const stocks=D.stocks;
  const datasets=Object.entries(SC).map(([sector,color])=>{
    const pts=stocks.filter(s=>s.sector===sector);
    if(!pts.length)return null;
    return{label:sector,
      data:pts.map(s=>({
        x:s.vol_annual!=null?+(s.vol_annual*100).toFixed(1):null,
        y:s.cagr_5y!=null?+(s.cagr_5y*100).toFixed(1):null,
        r:Math.max(5,(s.weight||0.03)*120),
        ticker:s.ticker,score:s.composite_score,weight:s.weight
      })).filter(d=>d.x!==null&&d.y!==null),
      backgroundColor:h2r(color,0.8),borderColor:color,borderWidth:1
    };
  }).filter(Boolean);

  new Chart(document.getElementById("cScatter"),{
    type:"bubble",data:{datasets},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{...chartBase.plugins,
        legend:{display:true,labels:{color:"#94a3b8",font:{size:10},boxWidth:10}},
        tooltip:{...chartBase.plugins.tooltip,callbacks:{label:ctx=>{
          const d=ctx.raw;
          return ` ${d.ticker} | Vol: ${d.x}% | CAGR: ${d.y}% | Score: ${d.score} | Peso: ${(d.weight*100).toFixed(1)}%`;
        }}}
      },
      scales:{...chartBase.scales,
        x:{...chartBase.scales.x,
           title:{display:true,text:"Volatilità annua (%)",color:"#475569",font:{size:11}},
           ticks:{...chartBase.scales.x.ticks,callback:v=>v+"%"}},
        y:{...chartBase.scales.y,
           title:{display:true,text:"CAGR 5 anni (%)",color:"#475569",font:{size:11}},
           ticks:{...chartBase.scales.y.ticks,callback:v=>v+"%"}}
      }}
  });

  document.getElementById("scoreBars").innerHTML=stocks.map(s=>{
    const color=sc(s.sector);
    const sColor=s.composite_score>=80?"#34D399":s.composite_score>=65?"#60A5FA":"#FBBF24";
    const bar=(v,c)=>`<div class="sbar-bg"><div class="sbar-fill" style="width:${v}%;background:${c}"></div></div>`;
    return `<div style="display:grid;grid-template-columns:90px 1fr 1fr 1fr;gap:12px;align-items:center;margin-bottom:14px">
      <span style="font-size:12px;font-weight:700;color:${color}">${s.ticker}</span>
      <div><div style="font-size:10px;color:var(--muted);margin-bottom:3px">Composito</div>
        <div class="sbar-wrap">${bar(s.composite_score,"var(--green)")}
          <span style="font-size:11px;color:${sColor};font-weight:700;min-width:28px">${s.composite_score.toFixed(0)}</span></div></div>
      <div><div style="font-size:10px;color:var(--muted);margin-bottom:3px">Tecnico</div>
        <div class="sbar-wrap">${bar(s.technical_score,"#60A5FA")}
          <span style="font-size:11px;color:#60A5FA;font-weight:700;min-width:28px">${s.technical_score.toFixed(0)}</span></div></div>
      <div><div style="font-size:10px;color:var(--muted);margin-bottom:3px">Fondamentale</div>
        <div class="sbar-wrap">${bar(s.fundamental_score,"#FBBF24")}
          <span style="font-size:11px;color:#FBBF24;font-weight:700;min-width:28px">${s.fundamental_score.toFixed(0)}</span></div></div>
    </div>`;
  }).join("");
}

// ── TAB 4: Titoli ─────────────────────────────────────────────────────────
let _sk="composite_score",_sd=-1;
function srt(k){if(_sk===k)_sd*=-1;else{_sk=k;_sd=-1;}renderTbl();}
function renderTbl(){
  const rows=[...D.stocks].sort((a,b)=>{
    const av=a[_sk],bv=b[_sk];
    if(av==null&&bv==null)return 0;if(av==null)return 1;if(bv==null)return-1;
    return(av>bv?1:-1)*_sd;
  });
  document.getElementById("stocksTbody").innerHTML=rows.map(s=>{
    const c=sc(s.sector);
    const scc=s.composite_score>=80?"#34D399":s.composite_score>=70?"#60A5FA":"#FBBF24";
    const vc=s.vol_annual>0.35?"var(--red)":s.vol_annual>0.22?"var(--yellow)":"var(--muted)";
    return `<tr>
      <td><span class="tbadge" style="background:${c}22;color:${c}">${s.ticker}</span></td>
      <td style="color:#94a3b8;max-width:140px">${s.name}</td>
      <td style="text-align:center"><span style="color:${scc};font-weight:800;font-size:14px">${s.composite_score.toFixed(1)}</span></td>
      <td style="text-align:right;font-weight:600">${pct(s.weight)}</td>
      <td style="text-align:right;color:var(--green);font-weight:600">${pct2(s.cagr_5y)}</td>
      <td style="text-align:right;color:${vc}">${pct(s.vol_annual)}</td>
      <td style="text-align:right;color:var(--red)">${pct(s.max_drawdown)}</td>
      <td style="text-align:right;color:${s.sharpe_ratio>1?"var(--green)":"var(--muted)"}">${num(s.sharpe_ratio)}</td>
      <td style="text-align:right;color:var(--muted)">${s.pe_ratio!=null?s.pe_ratio.toFixed(1):"—"}</td>
      <td style="text-align:right;color:var(--muted)">${s.roe!=null?(s.roe*100).toFixed(1)+"%":"—"}</td>
      <td style="text-align:right;color:var(--muted)">${pct2(s.revenue_growth)}</td>
      <td style="text-align:right;color:var(--muted)">${pct(s.fcf_yield)}</td>
      <td style="color:#64748b;font-size:11px;max-width:200px;white-space:normal">${s.rationale}</td>
    </tr>`;
  }).join("");
}
function b4(){renderTbl();}

// ── TAB 5: Notizie ────────────────────────────────────────────────────────
function b5(){
  const news   = D.news || {};
  const stocks = D.stocks;

  let pos=0, neu=0, neg=0;
  stocks.forEach(s=>{
    const n = news[s.ticker]; if(!n) return;
    if(n.signal==="POSITIVO") pos++;
    else if(n.signal==="NEGATIVO") neg++;
    else neu++;
  });

  const hasNews = pos+neu+neg > 0;

  document.getElementById("newsSummaryBar").innerHTML = `
    <div style="background:#052e16;border:1px solid #166534;border-radius:8px;padding:12px 20px;display:flex;align-items:center;gap:10px">
      <span style="font-size:22px">🟢</span>
      <div><div style="color:#4ade80;font-size:22px;font-weight:700">${pos}</div>
      <div style="color:#64748b;font-size:11px">POSITIVI</div></div>
    </div>
    <div style="background:#0c1a2e;border:1px solid #334155;border-radius:8px;padding:12px 20px;display:flex;align-items:center;gap:10px">
      <span style="font-size:22px">⚪</span>
      <div><div style="color:#94a3b8;font-size:22px;font-weight:700">${neu}</div>
      <div style="color:#64748b;font-size:11px">NEUTRI</div></div>
    </div>
    <div style="background:#2d0a0a;border:1px solid #7f1d1d;border-radius:8px;padding:12px 20px;display:flex;align-items:center;gap:10px">
      <span style="font-size:22px">🔴</span>
      <div><div style="color:#f87171;font-size:22px;font-weight:700">${neg}</div>
      <div style="color:#64748b;font-size:11px">NEGATIVI</div></div>
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:12px 20px;flex:1;min-width:220px">
      <div style="color:#94a3b8;font-size:11px;margin-bottom:6px">IMPATTO SUL PORTAFOGLIO</div>
      <div style="color:#e2e8f0;font-size:12px;line-height:1.5">
        Le notizie pesano il <b>10%</b> dello score composito.<br>
        Titoli con sentiment molto negativo e score &lt;55 vengono esclusi.<br>
        ${!hasNews ? '<span style="color:#fbbf24">⚠️ Notizie non disponibili — verifica connessione</span>' : ''}
      </div>
    </div>`;

  const grid = document.getElementById("newsGrid");
  grid.innerHTML = stocks.map(s=>{
    const n = news[s.ticker] || {
      signal:"NEUTRO", sentiment:"Neutro", news_score:50,
      news_raw_score:0, article_count:0, summary:"Nessun dato disponibile",
      key_points:[], risk_flags:[], opportunity:[], timing:"", favorable:null
    };
    const sigColor  = n.signal==="POSITIVO"?"#4ade80":n.signal==="NEGATIVO"?"#f87171":"#94a3b8";
    const sigBg     = n.signal==="POSITIVO"?"#052e16":n.signal==="NEGATIVO"?"#2d0a0a":"#0f172a";
    const sigBorder = n.signal==="POSITIVO"?"#166534":n.signal==="NEGATIVO"?"#7f1d1d":"#334155";
    const sigIcon   = n.signal==="POSITIVO"?"🟢":n.signal==="NEGATIVO"?"🔴":"⚪";

    // Bar: news_raw_score -100..+100 → percentuale 0..100
    const rawScore   = n.news_raw_score || 0;
    const barPct     = ((rawScore + 100) / 2);
    const scoreColor = rawScore>=20?"#4ade80":rawScore<=-20?"#f87171":"#94a3b8";
    const rawLabel   = rawScore > 0 ? "+"+rawScore : ""+rawScore;

    const keyPoints = (n.key_points||[]).map(p=>
      `<li style="color:#94a3b8;font-size:11px;margin-bottom:3px;line-height:1.4">• ${p}</li>`
    ).join("");

    const riskFlags = (n.risk_flags||[]).map(r=>
      `<span style="background:#3f0f0f;color:#f87171;font-size:10px;padding:2px 8px;border-radius:10px;margin-right:4px;margin-bottom:3px;display:inline-block">⚠ ${r}</span>`
    ).join("");

    const opportunities = (n.opportunity||[]).map(o=>
      `<span style="background:#052e16;color:#4ade80;font-size:10px;padding:2px 8px;border-radius:10px;margin-right:4px;margin-bottom:3px;display:inline-block">✓ ${o}</span>`
    ).join("");

    const favorTag = n.favorable === true
      ? `<span style="background:#052e16;border:1px solid #166534;color:#4ade80;font-size:10px;padding:2px 9px;border-radius:8px">✅ Momento favorevole</span>`
      : n.favorable === false
      ? `<span style="background:#2d0a0a;border:1px solid #7f1d1d;color:#f87171;font-size:10px;padding:2px 9px;border-radius:8px">❌ Momento sfavorevole</span>`
      : `<span style="background:#1e293b;color:#94a3b8;font-size:10px;padding:2px 9px;border-radius:8px">— Neutro</span>`;

    return `<div style="background:#0f172a;border:1px solid ${sigBorder};border-radius:12px;padding:16px;display:flex;flex-direction:column;gap:8px">

      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div>
          <span style="font-weight:700;color:#e2e8f0;font-size:15px">${s.ticker}</span>
          <span style="color:#475569;font-size:11px;margin-left:8px">${(s.name||"").substring(0,24)}</span>
        </div>
        <div style="background:${sigBg};border:1px solid ${sigBorder};border-radius:6px;padding:3px 10px;font-size:11px;color:${sigColor};font-weight:600">
          ${sigIcon} ${n.sentiment}
        </div>
      </div>

      <div style="display:flex;align-items:center;gap:8px">
        <div style="flex:1;height:7px;background:#1e293b;border-radius:4px;overflow:hidden">
          <div style="width:${barPct.toFixed(0)}%;height:100%;background:${scoreColor};border-radius:4px;transition:width 0.4s"></div>
        </div>
        <span style="color:${scoreColor};font-weight:700;font-size:12px;min-width:36px;text-align:right">${rawLabel}</span>
      </div>

      ${n.summary ? `<div style="color:#64748b;font-size:11px;line-height:1.5;font-style:italic">${n.summary}</div>` : ""}

      ${keyPoints ? `<ul style="margin:0;padding:0;list-style:none">${keyPoints}</ul>` : ""}

      ${riskFlags ? `<div style="display:flex;flex-wrap:wrap;gap:2px">${riskFlags}</div>` : ""}
      ${opportunities ? `<div style="display:flex;flex-wrap:wrap;gap:2px">${opportunities}</div>` : ""}

      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;flex-wrap:wrap;gap:6px">
        ${favorTag}
        <span style="color:#475569;font-size:10px">${n.article_count} articoli analizzati</span>
      </div>

      ${n.timing ? `<div style="color:#fbbf24;font-size:10px;border-top:1px solid #1e293b;padding-top:6px">⏱ ${n.timing}</div>` : ""}
    </div>`;
  }).join("");
}

// ── TAB 6: Previsione Monte Carlo ─────────────────────────────────────────
function b6(){
  const fc=D.forecast;
  if(!fc||!fc.labels){
    document.getElementById("kpiGridFc").innerHTML=
      '<div class="kpi"><div class="lbl">Previsione</div><div class="val" style="color:var(--muted)">N/D</div><div class="sub">dati non disponibili</div></div>';
    return;
  }

  // KPI previsione
  const months=fc.forecast_months;
  const years=(months/12).toFixed(1);
  const finalMed=fc.final_median_pct;
  const kpis=[
    {l:"Rendimento mediano atteso",v:(finalMed>=0?"+":"")+finalMed.toFixed(1)+"%",
     s:`su ${months} mesi (${years} anni)`,c:finalMed>=0?"var(--green)":"var(--red)"},
    {l:"Probabilità rendimento positivo",v:(fc.pct_positive*100).toFixed(0)+"%",
     s:"simulazioni con gain finale",c:fc.pct_positive>0.6?"var(--green)":"var(--yellow)"},
    {l:"Probabilità di battere S&P 500",v:(fc.pct_beat_spy*100).toFixed(0)+"%",
     s:"vs storico S&P 500 atteso",c:fc.pct_beat_spy>0.5?"var(--green)":"var(--yellow)"},
    {l:"Simulazioni",v:fc.n_sims.toLocaleString(),s:"percorsi Monte Carlo",c:"var(--teal)"},
    {l:"CAGR usato",v:(fc.ann_ret_used*100).toFixed(1)+"%",s:"basato su backtest storico",c:"var(--blue)"},
    {l:"Volatilità usata",v:(fc.ann_vol_used*100).toFixed(1)+"%",s:"annualizzata storica",c:"var(--yellow)"},
  ];
  document.getElementById("kpiGridFc").innerHTML=kpis.map(k=>
    `<div class="kpi"><div class="lbl">${k.l}</div>
     <div class="val" style="color:${k.c};font-size:22px">${k.v}</div>
     <div class="sub">${k.s}</div></div>`).join("");

  document.getElementById("fcSubtitle").innerHTML=
    `Simulazione su <strong style="color:var(--blue)">${fc.n_sims.toLocaleString()} percorsi casuali</strong> ·
     Orizzonte: <strong style="color:var(--green)">${months} mesi</strong> ·
     Rendimento usato: <strong style="color:var(--yellow)">${(fc.ann_ret_used*100).toFixed(1)}%/anno</strong> ·
     Vol: <strong style="color:var(--yellow)">${(fc.ann_vol_used*100).toFixed(1)}%/anno</strong>`;

  // ── Grafico principale Monte Carlo ──
  new Chart(document.getElementById("cForecast"),{
    type:"line",
    data:{labels:fc.labels,datasets:[
      // Banda confidenza 90% (p5-p95) — area grigia
      {label:"Intervallo 90%",data:fc.p95,borderColor:"transparent",
       backgroundColor:h2r("#60A5FA",0.08),fill:"+1",pointRadius:0,tension:0.4},
      {label:"_p5",data:fc.p5,borderColor:"transparent",
       backgroundColor:h2r("#60A5FA",0.08),fill:false,pointRadius:0,tension:0.4},
      // Banda confidenza 50% (p25-p75) — area più scura
      {label:"Intervallo 50%",data:fc.p75,borderColor:"transparent",
       backgroundColor:h2r("#60A5FA",0.18),fill:"+1",pointRadius:0,tension:0.4},
      {label:"_p25",data:fc.p25,borderColor:"transparent",
       backgroundColor:h2r("#60A5FA",0.18),fill:false,pointRadius:0,tension:0.4},
      // Mediana portafoglio
      {label:"Mediana portafoglio",data:fc.p50,borderColor:"#60A5FA",
       backgroundColor:"transparent",fill:false,pointRadius:0,borderWidth:2.5,tension:0.4},
      // Caso pessimistico (p5)
      {label:"Scenario pessimistico (5°%)",data:fc.p5,borderColor:"#F87171",
       backgroundColor:"transparent",fill:false,pointRadius:0,borderWidth:1.2,
       borderDash:[4,3],tension:0.4},
      // Caso ottimistico (p95)
      {label:"Scenario ottimistico (95°%)",data:fc.p95,borderColor:"#34D399",
       backgroundColor:"transparent",fill:false,pointRadius:0,borderWidth:1.2,
       borderDash:[4,3],tension:0.4},
      // S&P 500 mediano atteso (riferimento)
      {label:"S&P 500 atteso (storico)",data:fc.spy_p50,borderColor:"#FBBF24",
       backgroundColor:"transparent",fill:false,pointRadius:0,borderWidth:1.5,
       borderDash:[6,3],tension:0.4},
    ]},
    options:{responsive:true,maintainAspectRatio:false,...chartBase,
      plugins:{...chartBase.plugins,
        legend:{display:true,labels:{
          color:"#94a3b8",font:{size:10},boxWidth:12,
          filter:item=>!item.text.startsWith("_") && item.text!=="Intervallo 90%" && item.text!=="Intervallo 50%"
            ? true : item.text==="Intervallo 90%" || item.text==="Intervallo 50%"
        }},
        tooltip:{...chartBase.plugins.tooltip,callbacks:{
          label:ctx=>{
            if(ctx.dataset.label.startsWith("_")) return null;
            const v=ctx.parsed.y;
            return ` ${ctx.dataset.label}: ${v>=0?"+":""}${v.toFixed(1)}%`;
          }
        }}
      },
      scales:{...chartBase.scales,
        x:{...chartBase.scales.x,ticks:{...chartBase.scales.x.ticks,maxTicksLimit:12}},
        y:{...chartBase.scales.y,ticks:{...chartBase.scales.y.ticks,callback:v=>(v>=0?"+":"")+v+"%"}}
      }
    }
  });

  // ── Grafico distribuzione finale (istogramma approssimato) ──
  const nBins=20;
  const p5v=fc.p5[fc.p5.length-1], p95v=fc.p95[fc.p95.length-1];
  const binW=(p95v-p5v)/nBins;
  // Usa una distribuzione normale approssimata centrata su mediana
  const med=fc.p50[fc.p50.length-1];
  const sigma=(p95v-p5v)/3.29; // sigma approssimata da range 90%
  const bins=Array.from({length:nBins+1},(_,i)=>p5v+i*binW);
  function normalPdf(x,mu,s){return Math.exp(-0.5*((x-mu)/s)**2)/(s*Math.sqrt(2*Math.PI));}
  const densities=bins.slice(0,-1).map((b,i)=>{
    const x=(b+bins[i+1])/2;
    return +(normalPdf(x,med,sigma)*binW*100).toFixed(2);
  });
  const binLabels=bins.slice(0,-1).map(b=>(b>=0?"+":"")+b.toFixed(0)+"%");
  const barColors=bins.slice(0,-1).map(b=>b<0?"rgba(248,113,113,0.7)":b>med*0.5?"rgba(52,211,153,0.7)":"rgba(96,165,250,0.7)");

  new Chart(document.getElementById("cFcDist"),{
    type:"bar",
    data:{labels:binLabels,datasets:[{
      label:"Probabilità",data:densities,
      backgroundColor:barColors,borderRadius:2,borderSkipped:false,
    }]},
    options:{responsive:true,maintainAspectRatio:false,...chartBase,
      plugins:{...chartBase.plugins,legend:{display:false}},
      scales:{...chartBase.scales,
        x:{...chartBase.scales.x,ticks:{...chartBase.scales.x.ticks,maxTicksLimit:8,font:{size:9}}},
        y:{...chartBase.scales.y,ticks:{...chartBase.scales.y.ticks,callback:v=>v+"%",font:{size:9}},
           title:{display:true,text:"Densità probabilità",color:"#475569",font:{size:9}}}
      }
    }
  });

  // ── Scenari testuali ──
  const pf=fc.pct_positive, pb=fc.pct_beat_spy;
  const scenarios=[
    {emoji:"😱",label:"Pessimistico (5° percentile)",val:fc.p5[fc.p5.length-1],color:"#F87171",
     desc:"Solo il 5% delle simulazioni finisce peggio di questo valore"},
    {emoji:"📉",label:"Conservativo (25° percentile)",val:fc.p25[fc.p25.length-1],color:"#FB923C",
     desc:"Quartile inferiore — scenario cauto"},
    {emoji:"📊",label:"Mediana attesa (50°)",val:med,color:"#60A5FA",
     desc:"Il risultato più probabile secondo il modello"},
    {emoji:"📈",label:"Ottimistico (75° percentile)",val:fc.p75[fc.p75.length-1],color:"#34D399",
     desc:"Quartile superiore — scenario favorevole"},
    {emoji:"🚀",label:"Best case (95° percentile)",val:fc.p95[fc.p95.length-1],color:"#818CF8",
     desc:"Solo il 5% delle simulazioni finisce meglio di questo valore"},
  ];
  document.getElementById("fcScenarios").innerHTML=scenarios.map(s=>{
    const vStr=(s.val>=0?"+":"")+s.val.toFixed(1)+"%";
    const capital=D.capital*(1+s.val/100);
    return `<div style="background:var(--bg1);border:1px solid var(--border);border-radius:8px;padding:12px 16px;display:flex;align-items:center;gap:14px">
      <span style="font-size:20px">${s.emoji}</span>
      <div style="flex:1">
        <div style="font-size:11px;color:var(--muted);margin-bottom:3px">${s.label}</div>
        <div style="font-size:11px;color:#475569">${s.desc}</div>
      </div>
      <div style="text-align:right">
        <div style="font-size:16px;font-weight:800;color:${s.color}">${vStr}</div>
        <div style="font-size:10px;color:var(--dim)">€${capital.toLocaleString("it-IT",{maximumFractionDigits:0})}</div>
      </div>
    </div>`;
  }).join("");
}

// ── INIT ──────────────────────────────────────────────────────────────────
built[0]=true; b0();
</script>
</body>
</html>"""
