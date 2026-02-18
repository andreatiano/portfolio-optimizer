"""
Analisi fondamentale dei titoli.

Valuta ogni azienda su:
- Valutazione (P/E, P/B, P/S)
- Redditività (ROE, ROA, margini)
- Qualità (FCF Yield, crescita utili)
- Solidità finanziaria (debito, liquidità)
- Crescita (revenue growth, earnings growth)
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("portfolio_optimizer")


class FundamentalAnalyzer:
    """Analizza i dati fondamentali e produce uno score per ogni titolo."""

    # Soglie di riferimento per la valutazione (mediane di mercato)
    BENCHMARKS = {
        "pe_fair":    20,    # P/E "giusto" per una crescita media
        "pe_growth":  30,    # P/E accettabile per alta crescita
        "roe_good":   0.15,  # ROE buono
        "roe_great":  0.25,  # ROE eccellente
        "fcf_good":   0.04,  # FCF Yield > 4% ottimo
        "debt_low":   0.5,   # D/E basso
        "debt_high":  2.0,   # D/E problematico
        "margin_good": 0.15, # Margine netto buono
        "rev_growth_good": 0.10,  # +10% crescita ricavi
        "earn_growth_good": 0.10, # +10% crescita utili
    }

    def __init__(self, ticker_info: Dict):
        self.info = ticker_info

    def analyze_all(self, tickers: List[str]) -> Dict[str, Dict]:
        """Analizza tutti i ticker e restituisce metriche fondamentali."""
        results = {}
        for ticker in tickers:
            try:
                info = self.info.get(ticker, {})
                results[ticker] = self._analyze_ticker(ticker, info)
            except Exception as e:
                logger.warning(f"Errore analisi fondamentale {ticker}: {e}")
                results[ticker] = {"fundamental_score": 40, "filters_passed": False}
        return results

    # ─── ANALISI SINGOLO TICKER ──────────────────────────────────

    def _analyze_ticker(self, ticker: str, info: dict) -> dict:
        m = {}

        def sf(v):
            """Converte in float in modo sicuro — gestisce stringhe, None, NaN."""
            if v is None:
                return None
            try:
                f = float(v)
                return None if (f != f) else f   # nan check
            except (TypeError, ValueError):
                return None

        # ── Raccolta dati grezzi ─────────────────────────────────
        m["pe"] = sf(info.get("pe_ratio"))
        m["forward_pe"] = sf(info.get("forward_pe"))
        m["pb"] = sf(info.get("pb_ratio"))
        m["ps"] = sf(info.get("ps_ratio"))
        m["roe"] = sf(info.get("roe"))
        m["roa"] = sf(info.get("roa"))
        m["profit_margin"] = sf(info.get("profit_margin"))
        m["gross_margin"] = sf(info.get("gross_margin"))
        m["operating_margin"] = sf(info.get("operating_margin"))
        m["revenue_growth"] = sf(info.get("revenue_growth"))
        m["earnings_growth"] = sf(info.get("earnings_growth"))
        m["fcf_yield"] = sf(info.get("fcf_yield"))
        m["debt_to_equity"] = sf(info.get("debt_to_equity"))
        m["current_ratio"] = sf(info.get("current_ratio"))
        m["dividend_yield"] = sf(info.get("dividend_yield"))
        m["market_cap"] = sf(info.get("market_cap"))
        m["beta"] = sf(info.get("beta"))
        m["sector"] = info.get("sector", "Unknown")
        m["name"] = info.get("name", ticker)

        # ── Filtri di esclusione ─────────────────────────────────
        m["filters_passed"] = self._apply_filters(m)

        # ── Score fondamentale ───────────────────────────────────
        m["fundamental_score"] = self._compute_fundamental_score(m)

        # ── Motivazione sintetica ────────────────────────────────
        m["rationale"] = self._build_rationale(ticker, m)

        return m

    # ─── FILTRI DI ESCLUSIONE ────────────────────────────────────

    def _apply_filters(self, m: dict) -> bool:
        """
        Filtri permissivi: esclude solo casi estremi.
        Molti titoli europei su yfinance hanno dati fondamentali parziali
        quindi non escludiamo per dati mancanti, solo per valori palesemente anomali.
        """
        # Market cap < 1 miliardo: micro-cap fuori scope
        mc = m.get("market_cap")
        if mc is not None and mc < 1e9:
            return False
        # P/E > 500: quasi certamente un dato errato di yfinance
        pe = m.get("pe")
        if pe is not None and pe > 500:
            return False
        # D/E > 1000% (10x): leva finanziaria estrema (escludi solo casi limite)
        dte = m.get("debt_to_equity")
        if dte is not None and dte > 1000:
            return False
        return True

    # ─── SCORE FONDAMENTALE ──────────────────────────────────────

    def _compute_fundamental_score(self, m: dict) -> float:
        """
        Score fondamentale da 0 a 100.
        
        Componenti:
        - Valutazione relativa (20 pt)
        - Redditività / ROE (25 pt)
        - Qualità FCF (20 pt)
        - Crescita (20 pt)
        - Solidità finanziaria (15 pt)
        """
        score = 50.0
        b = self.BENCHMARKS

        # ── 1. Valutazione (P/E-based) ───────────────────────────
        pe = m.get("pe") or m.get("forward_pe")
        eg = m.get("earnings_growth") or 0
        if pe is not None and pe > 0:
            # PEG ratio approssimato
            peg = pe / (eg * 100 + 1e-6) if eg > 0 else pe / 15
            if peg < 1.0:   score += 12
            elif peg < 1.5: score += 6
            elif peg < 2.5: score += 0
            elif peg > 4.0: score -= 10
            elif peg > 3.0: score -= 5

        # ── 2. Redditività (ROE) ─────────────────────────────────
        roe = m.get("roe")
        if roe is not None:
            if roe >= b["roe_great"]:  score += 15
            elif roe >= b["roe_good"]: score += 8
            elif roe >= 0.08:          score += 3
            elif roe < 0:              score -= 12
            elif roe < 0.05:           score -= 5

        # ── 3. Margini di profitto ───────────────────────────────
        margin = m.get("profit_margin") or m.get("operating_margin")
        if margin is not None:
            if margin >= 0.25:   score += 8
            elif margin >= 0.15: score += 4
            elif margin >= 0.08: score += 1
            elif margin < 0:     score -= 10
            elif margin < 0.03:  score -= 4

        # ── 4. FCF Yield ─────────────────────────────────────────
        fcf = m.get("fcf_yield")
        if fcf is not None:
            if fcf >= b["fcf_good"]:    score += 12
            elif fcf >= 0.02:           score += 6
            elif fcf >= 0.01:           score += 2
            elif fcf < -0.01:           score -= 8

        # ── 5. Crescita ricavi ───────────────────────────────────
        rg = m.get("revenue_growth")
        if rg is not None:
            if rg >= 0.20:   score += 10
            elif rg >= 0.10: score += 6
            elif rg >= 0.05: score += 2
            elif rg < -0.05: score -= 8

        # ── 6. Crescita utili ────────────────────────────────────
        if eg > 0.20:   score += 8
        elif eg > 0.10: score += 4
        elif eg < -0.10: score -= 8

        # ── 7. Solidità finanziaria ──────────────────────────────
        dte = m.get("debt_to_equity")
        if dte is not None:
            dte_normalized = dte / 100  # yfinance usa percentuale
            if dte_normalized < b["debt_low"]:  score += 8
            elif dte_normalized < 1.0:          score += 3
            elif dte_normalized > b["debt_high"]: score -= 8

        cr = m.get("current_ratio")
        if cr is not None:
            if cr >= 2.0:   score += 3
            elif cr >= 1.5: score += 1
            elif cr < 1.0:  score -= 5

        return max(0.0, min(100.0, score))

    # ─── MOTIVAZIONE ─────────────────────────────────────────────

    def _build_rationale(self, ticker: str, m: dict) -> str:
        """Genera una motivazione leggibile per la selezione del titolo."""
        parts = []
        name = m.get("name", ticker)
        sector = m.get("sector", "")

        if sector and sector != "Unknown":
            parts.append(f"Settore {sector}")

        roe = m.get("roe")
        if roe and roe >= 0.15:
            parts.append(f"ROE elevato ({roe:.0%})")

        pe = m.get("pe")
        if pe and 5 < pe < 30:
            parts.append(f"Valutazione ragionevole (P/E {pe:.1f}x)")
        elif pe and 30 <= pe <= 50:
            parts.append(f"P/E elevato giustificato dalla crescita ({pe:.1f}x)")

        eg = m.get("earnings_growth")
        if eg and eg >= 0.10:
            parts.append(f"Crescita utili solida (+{eg:.0%})")

        fcf = m.get("fcf_yield")
        if fcf and fcf >= 0.03:
            parts.append(f"FCF Yield attraente ({fcf:.1%})")

        dte = m.get("debt_to_equity")
        if dte is not None and dte / 100 < 0.5:
            parts.append("basso indebitamento")

        if not parts:
            return f"Azienda di qualità nel settore {sector or 'diversificato'}"

        return "; ".join(parts) + "."
