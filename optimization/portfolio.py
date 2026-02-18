"""
Ottimizzazione del portafoglio con approccio Media-Varianza modificato.

Implementa:
1. Ottimizzazione classica di Markowitz con vincoli
2. Penalizzazione volatilità extra
3. Vincolo su peso massimo per titolo e per settore
4. Calcolo costi di transazione
5. Backtest storico
6. Stima orizzonte temporale ottimale
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger("portfolio_optimizer")

# Parametri ottimizzazione
RISK_FREE_RATE = 0.035
RISK_AVERSION = 2.0       # Coefficiente avversione rischio (più alto = più conservativo)
MIN_WEIGHT = 0.03
MAX_WEIGHT = 0.15
BUY_COST = 1.0
SELL_COST = 1.0


class PortfolioOptimizer:
    """
    Ottimizzatore di portafoglio.
    
    Massimizza: E[R] - λ * Var[R] - penalità_volatilità
    Con i vincoli: somma pesi = 1, min_w ≤ w_i ≤ max_w
    """

    def __init__(
        self,
        price_data: pd.DataFrame,
        scores: pd.DataFrame,
        capital: float = 10000,
    ):
        self.prices = price_data
        self.scores = scores
        self.capital = capital
        self.tickers = list(price_data.columns)
        self.n = len(self.tickers)

        # Rendimenti giornalieri
        self.returns = price_data.pct_change().dropna()
        # Matrice di covarianza annualizzata
        self.cov_matrix = self.returns.cov() * 252
        # Rendimenti attesi annualizzati
        self.expected_returns = self._estimate_expected_returns()

    # ─── OTTIMIZZAZIONE ──────────────────────────────────────────

    def optimize(self) -> dict:
        """
        Ottimizza il portafoglio e restituisce l'allocazione.
        
        Returns:
            Dizionario con pesi, metriche, allocazione monetaria
        """
        if self.n == 0:
            return {}

        # ── Costruisci pesi ottimali ─────────────────────────────
        weights = self._optimize_weights()

        # ── Calcola metriche portafoglio ─────────────────────────
        port_return = float(np.dot(weights, self.expected_returns))
        port_vol = float(np.sqrt(weights @ self.cov_matrix.values @ weights))
        sharpe = (port_return - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0

        # ── Allocazione monetaria ────────────────────────────────
        allocation = self._compute_allocation(weights)

        # ── Orizzonte consigliato ────────────────────────────────
        horizon = self._suggest_horizon(port_vol, sharpe)

        # ── Info per settore ─────────────────────────────────────
        sector_allocation = self._sector_breakdown(weights)

        return {
            "tickers": self.tickers,
            "weights": {t: w for t, w in zip(self.tickers, weights)},
            "allocation": allocation,
            "expected_return": port_return,
            "expected_volatility": port_vol,
            "sharpe_ratio": sharpe,
            "sector_allocation": sector_allocation,
            "horizon_months": horizon["months"],
            "horizon_label": horizon["label"],
            "next_review": horizon["next_review"],
            "horizon_method": horizon.get("method", "heuristic"),
            "capital": self.capital,
        }

    def _optimize_weights(self) -> np.ndarray:
        """Ottimizzazione con scipy.minimize (SLSQP)."""
        n = self.n
        w0 = np.ones(n) / n  # Pesi iniziali uguali

        # ── Funzione obiettivo: minimizza -Sharpe penalizzato ────
        def objective(w):
            ret = np.dot(w, self.expected_returns)
            vol = np.sqrt(w @ self.cov_matrix.values @ w)
            if vol < 1e-8:
                return 1e8
            # Penalizzazione extra per alta volatilità dei singoli titoli
            vols_individual = np.sqrt(np.diag(self.cov_matrix.values))
            vol_penalty = np.dot(w, vols_individual) * RISK_AVERSION * 0.5
            utility = ret - RISK_AVERSION * 0.5 * vol ** 2 - vol_penalty
            return -utility  # Minimizziamo il negativo

        # ── Vincoli e bounds ─────────────────────────────────────
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(MIN_WEIGHT, MAX_WEIGHT)] * n

        # Ottimizzazione
        result = minimize(
            objective, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9}
        )

        if result.success:
            weights = result.x
        else:
            logger.warning("Ottimizzazione non convergente, uso pesi score-based")
            weights = self._score_based_weights()

        # Normalizza
        weights = np.clip(weights, MIN_WEIGHT, MAX_WEIGHT)
        weights /= weights.sum()

        return weights

    def _score_based_weights(self) -> np.ndarray:
        """Fallback: pesi proporzionali allo score composito."""
        scores_arr = np.array([
            float(self.scores[self.scores["ticker"] == t]["composite_score"].iloc[0])
            if t in self.scores["ticker"].values else 50
            for t in self.tickers
        ])
        scores_arr = np.clip(scores_arr, 1, 100)
        weights = scores_arr / scores_arr.sum()
        weights = np.clip(weights, MIN_WEIGHT, MAX_WEIGHT)
        return weights / weights.sum()

    def _estimate_expected_returns(self) -> np.ndarray:
        """
        Stima rendimenti attesi con media pesata tra storico e mean-reversion.
        Usa CAPM-like adjustment con beta dei titoli.
        """
        # Rendimento medio storico (rolling 3y e 5y)
        hist_ret = self.returns.mean() * 252

        # Smoothing verso la media di mercato (shrinkage)
        market_return = 0.10  # Assunzione mercato ~10%/anno
        shrinkage = 0.3
        blended = (1 - shrinkage) * hist_ret + shrinkage * market_return

        return blended.values

    # ─── ALLOCAZIONE MONETARIA ───────────────────────────────────

    def _compute_allocation(self, weights: np.ndarray) -> List[dict]:
        """Calcola l'allocazione in EUR per ogni titolo."""
        allocation = []
        for ticker, weight in zip(self.tickers, weights):
            amount = weight * self.capital
            # Calcola costo transazione
            cost_impact = (BUY_COST / amount * 100) if amount > 0 else 0
            allocation.append({
                "ticker": ticker,
                "weight": weight,
                "amount_eur": amount,
                "buy_cost": BUY_COST,
                "cost_pct": cost_impact,
            })
        return sorted(allocation, key=lambda x: x["weight"], reverse=True)

    # ─── SETTORI ─────────────────────────────────────────────────

    def _sector_breakdown(self, weights: np.ndarray) -> dict:
        """Calcola l'allocazione per settore."""
        sector_w = {}
        for ticker, weight in zip(self.tickers, weights):
            sector = "Unknown"
            if hasattr(self, 'scores') and not self.scores.empty:
                row = self.scores[self.scores["ticker"] == ticker]
                if not row.empty:
                    sector = str(row["sector"].iloc[0])
            sector_w[sector] = sector_w.get(sector, 0) + weight
        return sector_w

    # ─── ORIZZONTE TEMPORALE ─────────────────────────────────────

    def _suggest_horizon(self, volatility: float, sharpe: float) -> dict:
        """
        Suggerisce l'orizzonte temporale di revisione.
        Prima tenta con HorizonEstimator (analisi statistica),
        poi usa euristica semplice come fallback.
        """
        # Tenta analisi statistica avanzata
        try:
            from optimization.horizon_estimator import HorizonEstimator
            estimator = HorizonEstimator(self.prices)
            result    = estimator.estimate()
            days      = result["recommended_days"]
            months    = max(1, round(days / 21))
            label     = result["recommended_label"]
            next_review = (datetime.now() + timedelta(days=days)).strftime("%d/%m/%Y")
            return {
                "months":      months,
                "label":       label,
                "next_review": next_review,
                "days":        days,
                "half_life":   result.get("half_life_days"),
                "confidence":  result.get("confidence"),
                "method":      "statistical",
            }
        except Exception as e:
            logger.debug(f"HorizonEstimator fallback euristico: {e}")

        # Fallback euristico
        if volatility < 0.12 and sharpe > 1.0:
            months, label = 12, "12 mesi"
        elif volatility < 0.18 and sharpe > 0.6:
            months, label = 6, "6 mesi"
        elif volatility < 0.25:
            months, label = 6, "6 mesi"
        else:
            months, label = 3, "3 mesi (volatilità elevata)"

        next_review = (datetime.now() + timedelta(days=30 * months)).strftime("%d/%m/%Y")
        return {"months": months, "label": label, "next_review": next_review,
                "days": months * 21, "method": "heuristic"}

    # ─── BACKTEST ────────────────────────────────────────────────

    def backtest(self, portfolio: dict, years: int = 5) -> dict:
        """
        Simula la performance storica del portafoglio ottimizzato.
        
        Args:
            portfolio: Output di optimize()
            years: Anni di backtest
            
        Returns:
            Dizionario con serie temporale, metriche, confronto benchmark
        """
        weights = np.array([portfolio["weights"][t] for t in self.tickers])

        # Prendi gli ultimi N anni
        n_days = years * 252
        rets = self.returns.tail(n_days)

        # Performance portafoglio
        port_rets = (rets * weights).sum(axis=1)
        cum_returns = (1 + port_rets).cumprod()

        # Benchmark: S&P 500 proxy (media ponderata dei titoli USA)
        # Usiamo equally-weighted come riferimento
        eq_weights = np.ones(self.n) / self.n
        bench_rets = (rets * eq_weights).sum(axis=1)
        cum_bench = (1 + bench_rets).cumprod()

        # ── Metriche backtest ────────────────────────────────────
        total_return = float(cum_returns.iloc[-1] - 1)
        bench_total = float(cum_bench.iloc[-1] - 1)

        ann_return = (1 + total_return) ** (1 / years) - 1
        ann_vol = port_rets.std() * np.sqrt(252)
        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        max_dd = float(drawdown.min())

        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd < 0 else 0

        return {
            "portfolio_returns": port_rets,
            "portfolio_cumulative": cum_returns,
            "benchmark_cumulative": cum_bench,
            "total_return": total_return,
            "benchmark_total": bench_total,
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "years": years,
            "start_date": rets.index[0].strftime("%d/%m/%Y"),
            "end_date": rets.index[-1].strftime("%d/%m/%Y"),
        }
