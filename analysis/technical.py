"""
Analisi tecnica e statistica dei prezzi storici.

Calcola per ogni titolo:
- Volatilità annualizzata
- Max Drawdown
- Trend di crescita (CAGR)
- Consistenza della crescita
- Sharpe Ratio semplificato
- Score di stabilità
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("portfolio_optimizer")

RISK_FREE_RATE = 0.035  # 3.5% annuo


class TechnicalAnalyzer:
    """
    Analizza i prezzi storici di ogni titolo per estrarre
    metriche di rischio, trend e stabilità.
    """

    def __init__(self, price_data: pd.DataFrame):
        """
        Args:
            price_data: DataFrame con prezzi di chiusura, index=date, cols=ticker
        """
        self.prices = price_data
        # Rendimenti giornalieri
        self.returns = price_data.pct_change().dropna()

    def analyze_all(self, tickers: List[str]) -> Dict[str, Dict]:
        """Analizza tutti i ticker e restituisce un dizionario di metriche."""
        results = {}
        valid_tickers = [t for t in tickers if t in self.prices.columns]

        for ticker in valid_tickers:
            try:
                results[ticker] = self._analyze_ticker(ticker)
            except Exception as e:
                logger.warning(f"Errore analisi tecnica {ticker}: {e}")
                # Fallback: calcola almeno le metriche base senza resample
                try:
                    results[ticker] = self._analyze_ticker_safe(ticker)
                except Exception:
                    results[ticker] = self._empty_metrics(ticker)

        return results

    # ─── ANALISI SINGOLO TICKER ──────────────────────────────────

    def _analyze_ticker(self, ticker: str) -> dict:
        prices = self.prices[ticker].dropna()
        rets = self.returns[ticker].dropna()

        if len(prices) < 252:
            return self._empty_metrics(ticker)

        metrics = {}

        # ── Rendimenti ──────────────────────────────────────────
        metrics["cagr_3y"] = self._cagr(prices, years=3)
        metrics["cagr_5y"] = self._cagr(prices, years=5)
        metrics["cagr_10y"] = self._cagr(prices, years=10)

        # ── Volatilità ──────────────────────────────────────────
        metrics["vol_annual"] = rets.std() * np.sqrt(252)
        metrics["vol_3y"] = rets.tail(252*3).std() * np.sqrt(252)
        metrics["vol_1y"] = rets.tail(252).std() * np.sqrt(252)

        # ── Drawdown ────────────────────────────────────────────
        metrics["max_drawdown"] = self._max_drawdown(prices)
        metrics["max_drawdown_3y"] = self._max_drawdown(prices.tail(252*3))
        metrics["avg_drawdown"] = self._avg_drawdown(prices)

        # ── Sharpe Ratio ────────────────────────────────────────
        avg_ret = rets.mean() * 252
        vol = metrics["vol_annual"]
        metrics["sharpe_ratio"] = (avg_ret - RISK_FREE_RATE) / vol if vol > 0 else 0
        metrics["sharpe_3y"] = self._sharpe_period(rets.tail(252*3))

        # ── Trend e stabilità ───────────────────────────────────
        metrics["trend_r2"] = self._trend_quality(prices)
        metrics["trend_r2_3y"] = self._trend_quality(prices.tail(252*3))
        metrics["consistency_score"] = self._consistency_score(rets)
        metrics["positive_years_pct"] = self._positive_years_pct(rets)

        # ── Skewness / Tail risk ─────────────────────────────────
        metrics["skewness"] = float(rets.skew())
        metrics["kurtosis"] = float(rets.kurtosis())
        metrics["var_95"] = float(rets.quantile(0.05))  # VaR 95%
        metrics["cvar_95"] = float(rets[rets <= metrics["var_95"]].mean())  # CVaR

        # ── Score tecnico aggregato ──────────────────────────────
        metrics["technical_score"] = self._compute_technical_score(metrics)

        return metrics

    # ─── METRICHE CORE ───────────────────────────────────────────

    def _cagr(self, prices: pd.Series, years: int) -> float:
        """Compound Annual Growth Rate."""
        n = min(years * 252, len(prices))
        if n < 60:
            return np.nan
        try:
            p_start = float(prices.iloc[-n])
            p_end   = float(prices.iloc[-1])
        except (TypeError, ValueError):
            return np.nan
        if p_start <= 0 or np.isnan(p_start) or np.isnan(p_end):
            return np.nan
        actual_years = n / 252
        return (p_end / p_start) ** (1 / actual_years) - 1

    def _max_drawdown(self, prices: pd.Series) -> float:
        """Maximum Drawdown come numero negativo."""
        if len(prices) < 2:
            return 0.0
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return float(drawdown.min())

    def _avg_drawdown(self, prices: pd.Series) -> float:
        """Drawdown medio (media dei drawdown negativi)."""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        neg = drawdown[drawdown < -0.02]
        return float(neg.mean()) if len(neg) > 0 else 0.0

    def _sharpe_period(self, rets: pd.Series) -> float:
        """Sharpe Ratio per un periodo specifico."""
        if len(rets) < 60:
            return 0.0
        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)
        return (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    def _trend_quality(self, prices: pd.Series) -> float:
        """
        R² della regressione log-lineare sui prezzi.
        Alto R² = crescita costante e lineare in scala log.
        """
        if len(prices) < 60:
            return 0.0
        y = np.log(prices.values)
        x = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return max(0.0, r_value ** 2)

    def _consistency_score(self, rets: pd.Series) -> float:
        """
        Percentuale di trimestri con rendimento positivo.
        Misura la consistenza della crescita.
        """
        # Compatibilità pandas <2.2 (Q) e >=2.2 (QE)
        for freq in ("QE", "Q"):
            try:
                quarterly = rets.resample(freq).apply(lambda x: (1 + x).prod() - 1)
                break
            except Exception:
                continue
        else:
            return 0.5
        if len(quarterly) < 4:
            return 0.5
        return float((quarterly > 0).mean())

    def _positive_years_pct(self, rets: pd.Series) -> float:
        """Percentuale di anni con rendimento positivo."""
        for freq in ("YE", "Y"):
            try:
                annual = rets.resample(freq).apply(lambda x: (1 + x).prod() - 1)
                break
            except Exception:
                continue
        else:
            return 0.5
        if len(annual) < 2:
            return 0.5
        return float((annual > 0).mean())

    # ─── SCORE AGGREGATO ─────────────────────────────────────────

    def _compute_technical_score(self, m: dict) -> float:
        """
        Score da 0 a 100 basato sulle metriche tecniche.
        Premia: CAGR alto, bassa volatilità, basso drawdown, trend costante.
        Penalizza: alta volatilità, drawdown profondo, crescita inconsistente.
        """
        score = 50.0  # Base

        # ── CAGR 5Y (peso alto) ─────────────────────────────────
        cagr5 = m.get("cagr_5y")
        if cagr5 is not None and not np.isnan(cagr5):
            if cagr5 > 0.20:   score += 15
            elif cagr5 > 0.12: score += 10
            elif cagr5 > 0.07: score += 5
            elif cagr5 < 0:    score -= 15

        # ── Volatilità (penalizzazione) ──────────────────────────
        vol = m.get("vol_annual", 0.3)
        if vol < 0.15:   score += 12
        elif vol < 0.22: score += 6
        elif vol > 0.40: score -= 15
        elif vol > 0.30: score -= 7

        # ── Max Drawdown (penalizzazione forte) ──────────────────
        mdd = m.get("max_drawdown", -0.3)
        if mdd > -0.20:  score += 10
        elif mdd > -0.35: score += 3
        elif mdd < -0.55: score -= 15
        elif mdd < -0.45: score -= 8

        # ── Trend quality (R²) ───────────────────────────────────
        r2 = m.get("trend_r2", 0.5)
        score += (r2 - 0.5) * 20  # da -10 a +10

        # ── Consistenza ─────────────────────────────────────────
        cons = m.get("consistency_score", 0.5)
        score += (cons - 0.5) * 16  # da -8 a +8

        # ── Sharpe ──────────────────────────────────────────────
        sharpe = m.get("sharpe_ratio", 0)
        if sharpe > 1.5:   score += 8
        elif sharpe > 0.8: score += 4
        elif sharpe < 0:   score -= 8

        return max(0.0, min(100.0, score))

    def _analyze_ticker_safe(self, ticker: str) -> dict:
        """
        Versione robusta di _analyze_ticker che non usa resample.
        Usata come fallback quando resample fallisce (es. versione pandas).
        """
        prices = self.prices[ticker].dropna()
        rets   = self.returns[ticker].dropna()

        if len(prices) < 60:
            return self._empty_metrics(ticker)

        metrics = {}
        metrics["cagr_3y"]  = self._cagr(prices, years=3)
        metrics["cagr_5y"]  = self._cagr(prices, years=5)
        metrics["cagr_10y"] = self._cagr(prices, years=10)

        vol = rets.std() * np.sqrt(252)
        metrics["vol_annual"]    = float(vol)
        metrics["vol_3y"]        = float(rets.tail(252*3).std() * np.sqrt(252))
        metrics["vol_1y"]        = float(rets.tail(252).std() * np.sqrt(252))
        metrics["max_drawdown"]  = self._max_drawdown(prices)
        metrics["max_drawdown_3y"] = self._max_drawdown(prices.tail(252*3))
        metrics["avg_drawdown"]  = self._avg_drawdown(prices)

        avg_ret = rets.mean() * 252
        metrics["sharpe_ratio"] = (avg_ret - RISK_FREE_RATE) / vol if vol > 0 else 0
        metrics["sharpe_3y"]    = self._sharpe_period(rets.tail(252*3))
        metrics["trend_r2"]     = self._trend_quality(prices)
        metrics["trend_r2_3y"]  = self._trend_quality(prices.tail(252*3))

        # Consistenza senza resample: % mesi positivi
        monthly_rets = rets.groupby([rets.index.year, rets.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        )
        metrics["consistency_score"]   = float((monthly_rets > 0).mean()) if len(monthly_rets) > 6 else 0.5
        metrics["positive_years_pct"]  = 0.6  # default neutro

        metrics["skewness"]  = float(rets.skew())
        metrics["kurtosis"]  = float(rets.kurtosis())
        metrics["var_95"]    = float(rets.quantile(0.05))
        metrics["cvar_95"]   = float(rets[rets <= metrics["var_95"]].mean()) if (rets <= metrics["var_95"]).any() else metrics["var_95"]

        metrics["technical_score"] = self._compute_technical_score(metrics)
        return metrics

    def _empty_metrics(self, ticker: str) -> dict:
        return {
            "cagr_3y": np.nan, "cagr_5y": np.nan, "cagr_10y": np.nan,
            "vol_annual": np.nan, "vol_3y": np.nan, "vol_1y": np.nan,
            "max_drawdown": np.nan, "max_drawdown_3y": np.nan, "avg_drawdown": np.nan,
            "sharpe_ratio": 0, "sharpe_3y": 0,
            "trend_r2": 0, "trend_r2_3y": 0,
            "consistency_score": 0, "positive_years_pct": 0,
            "skewness": 0, "kurtosis": 0,
            "var_95": np.nan, "cvar_95": np.nan,
            "technical_score": 30,
        }
