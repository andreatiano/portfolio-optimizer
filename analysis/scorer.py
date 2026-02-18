"""
Modulo di scoring e selezione titoli.

Score composito:
  40% Fondamentale  (P/E, ROE, FCF, crescita)
  35% Tecnico       (CAGR, volatilità, drawdown, trend)
  15% Momentum      (CAGR 3y vs 5y)
  10% News Sentiment (ultimi 14 giorni — opzionale, se disponibile)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("portfolio_optimizer")

# Pesi dello score finale
WEIGHTS = {
    "fundamental": 0.40,
    "technical":   0.35,
    "momentum":    0.15,
    "news":        0.10,   # attivato solo se news_scores disponibili
}

# Senza news: ridistribuisci il 10% proporzionalmente
WEIGHTS_NO_NEWS = {
    "fundamental": 0.45,
    "technical":   0.40,
    "momentum":    0.15,
}

MAX_PER_SECTOR  = 3
MAX_PER_COUNTRY = 4   # non troppa concentrazione su un singolo paese


class StockScorer:
    """Combina score tecnico, fondamentale e news sentiment."""

    def __init__(self, technical: Dict, fundamental: Dict,
                 ticker_info: Dict, news_scores: Optional[Dict] = None):
        self.tech  = technical
        self.fund  = fundamental
        self.info  = ticker_info
        self.news  = news_scores or {}   # dict {ticker: {news_score, signal, ...}}

    def rank_stocks(self) -> pd.DataFrame:
        rows       = []
        all_tickers = set(self.tech.keys()) & set(self.fund.keys())
        use_news   = bool(self.news)
        weights    = WEIGHTS if use_news else WEIGHTS_NO_NEWS

        for ticker in all_tickers:
            t    = self.tech[ticker]
            f    = self.fund[ticker]
            info = self.info.get(ticker, {})
            n    = self.news.get(ticker, {})

            # ── Filtro hard ──────────────────────────────────────
            if not f.get("filters_passed", True):
                continue
            vol   = t.get("vol_annual")
            cagr  = t.get("cagr_5y")
            ts_val = t.get("technical_score", 0)
            if vol is None and cagr is None and ts_val == 0:
                continue

            # ── Score componenti ─────────────────────────────────
            ts = float(t.get("technical_score", 50))
            fs = float(f.get("fundamental_score", 50))

            c3 = t.get("cagr_3y", np.nan)
            c5 = t.get("cagr_5y", np.nan)
            try:
                c3f, c5f = float(c3), float(c5)
                if not np.isnan(c3f) and not np.isnan(c5f) and c5f > 0:
                    momentum = min(100, max(0, 50 + (c3f - c5f) / c5f * 100))
                else:
                    momentum = 50.0
            except Exception:
                momentum = 50.0
                c3f = c5f = float("nan")

            # News score: il modulo restituisce -100..+100, normalizziamo a 0..100
            raw_ns = n.get("news_score", 0) if n else 0
            try:
                ns = (float(raw_ns) + 100) / 2   # -100->0, 0->50, +100->100
            except Exception:
                ns = 50.0

            # ── Composito ────────────────────────────────────────
            if use_news and n:
                composite = (
                    weights["fundamental"] * fs +
                    weights["technical"]   * ts +
                    weights["momentum"]    * momentum +
                    weights["news"]        * ns
                )
            else:
                composite = (
                    WEIGHTS_NO_NEWS["fundamental"] * fs +
                    WEIGHTS_NO_NEWS["technical"]   * ts +
                    WEIGHTS_NO_NEWS["momentum"]    * momentum
                )

            sentiment   = n.get("sentiment", "Neutro") if n else "Neutro"
            news_signal = ("POSITIVO" if sentiment == "Positivo"
                           else "NEGATIVO" if sentiment == "Negativo" else "NEUTRO")

            rows.append({
                "ticker":            ticker,
                "name":              info.get("name", ticker),
                "sector":            info.get("sector", "Unknown"),
                "country":           info.get("country", "Unknown"),
                "composite_score":   composite,
                "technical_score":   ts,
                "fundamental_score": fs,
                "momentum_score":    momentum,
                "news_score":        ns,
                "news_raw_score":    raw_ns,
                "news_signal":       news_signal,
                "news_sentiment":    sentiment,
                "news_summary":      n.get("summary", "") if n else "",
                "news_key_points":   n.get("key_points", []) if n else [],
                "news_risk_flags":   n.get("risk_flags", []) if n else [],
                "news_opportunity":  n.get("opportunity_flags", []) if n else [],
                "news_timing":       n.get("timing_note", "") if n else "",
                "news_favorable":    n.get("favorable") if n else None,
                "news_articles":     n.get("articles_count", 0) if n else 0,
                # Metriche tecniche/fondamentali
                "cagr_5y":           c5f if not np.isnan(c5f) else None,
                "cagr_3y":           c3f if not np.isnan(c3f) else None,
                "vol_annual":        t.get("vol_annual"),
                "max_drawdown":      t.get("max_drawdown"),
                "sharpe_ratio":      t.get("sharpe_ratio"),
                "pe_ratio":          f.get("pe"),
                "roe":               f.get("roe"),
                "revenue_growth":    f.get("revenue_growth"),
                "earnings_growth":   f.get("earnings_growth"),
                "fcf_yield":         f.get("fcf_yield"),
                "market_cap":        info.get("market_cap"),
                "beta":              info.get("beta"),
                "rationale":         f.get("rationale", ""),
                "currency":          info.get("currency", "USD"),
            })


        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df

    def select_top_stocks(
        self,
        ranked: pd.DataFrame,
        min_stocks: int = 8,
        max_stocks: int = 15,
    ) -> List[str]:
        if ranked.empty:
            return []

        selected     = []
        sector_count: Dict[str, int] = {}
        country_count: Dict[str, int] = {}

        candidates = ranked[ranked["composite_score"] >= 50].copy()

        for _, row in candidates.iterrows():
            if len(selected) >= max_stocks:
                break

            ticker  = row["ticker"]
            sector  = row["sector"]
            country = row["country"]

            if sector_count.get(sector, 0) >= MAX_PER_SECTOR:
                continue
            if country_count.get(country, 0) >= MAX_PER_COUNTRY:
                continue

            # Penalizza titoli con news molto negative: richiede score >= 55
            if row.get("news_signal") == "NEGATIVO" and row["composite_score"] < 55:
                logger.info(f"Saltato {ticker}: news negative e score <55")
                continue

            selected.append(ticker)
            sector_count[sector]   = sector_count.get(sector, 0) + 1
            country_count[country] = country_count.get(country, 0) + 1

        # Fallback: abbassa soglia se pochi titoli
        if len(selected) < min_stocks:
            for _, row in ranked.iterrows():
                if len(selected) >= min_stocks:
                    break
                if row["ticker"] not in selected:
                    selected.append(row["ticker"])

        logger.info(f"Selezionati {len(selected)} titoli | settori: {sector_count}")
        return selected
