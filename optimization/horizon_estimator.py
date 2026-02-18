"""
╔══════════════════════════════════════════════════════════════╗
║            HORIZON ESTIMATOR — v7                            ║
║      Stima automatica dell'orizzonte temporale ottimale      ║
╚══════════════════════════════════════════════════════════════╝

Invece di fissare l'orizzonte di revisione a priori, analizza
statisticamente la stabilità della strategia su diverse finestre
temporali e stima quella più coerente con i dati.

Metodi impiegati:
  1. Stabilità pesi  — std media dei pesi ottimali su rolling windows
  2. Sharpe OOS      — Sharpe ratio out-of-sample per ogni finestra
  3. Half-life       — Stima velocità di decadimento della stabilità (AR-1)
  4. Ljung-Box test  — Autocorrelazione rendimenti (proxy di prevedibilità)

Utilizzo:
    estimator = HorizonEstimator(price_data)
    result = estimator.estimate()
    print(f"Orizzonte consigliato: {result['recommended_days']} giorni")
    print(f"Half-life stimato:     {result['half_life_days']:.0f} giorni")
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger("portfolio_optimizer")

# Parametri ottimizzazione interna
RISK_FREE_RATE = 0.035
RISK_AVERSION  = 2.0
MIN_WEIGHT     = 0.03
MAX_WEIGHT     = 0.20


class HorizonEstimator:
    """
    Stima l'orizzonte temporale ottimale di revisione del portafoglio
    analizzando la stabilità statistica su diverse finestre candidate.

    Args:
        price_data:       DataFrame prezzi (index=date, cols=ticker)
        candidate_days:   Lista finestre candidate in giorni trading
        stability_weight: Peso stabilità pesi nello score composito (default 0.40)
        sharpe_weight:    Peso Sharpe OOS nello score composito (default 0.40)
        cost_weight:      Peso costo revisione nello score composito (default 0.20)
        min_days:         Orizzonte minimo (default 21 = 1 mese)
        max_days:         Orizzonte massimo (default 252 = 1 anno)
    """

    DEFAULT_CANDIDATES = [21, 42, 63, 126, 189, 252]  # 1m, 2m, 3m, 6m, 9m, 12m

    def __init__(
        self,
        price_data:       pd.DataFrame,
        candidate_days:   Optional[List[int]] = None,
        stability_weight: float = 0.40,
        sharpe_weight:    float = 0.40,
        cost_weight:      float = 0.20,
        min_days:         int   = 21,
        max_days:         int   = 252,
    ):
        self.prices     = price_data.dropna(axis=1, how="all")
        self.returns    = self.prices.pct_change().dropna()
        self.tickers    = list(self.prices.columns)
        self.n          = len(self.tickers)
        self.candidates = candidate_days or self.DEFAULT_CANDIDATES
        self.candidates = [c for c in self.candidates if min_days <= c <= max_days]

        # Pesi score composito
        total_w = stability_weight + sharpe_weight + cost_weight
        self.w_stability = stability_weight / total_w
        self.w_sharpe    = sharpe_weight    / total_w
        self.w_cost      = cost_weight      / total_w

        self.min_days = min_days
        self.max_days = max_days

    # ─── METODO PRINCIPALE ───────────────────────────────────────────────────

    def estimate(self) -> dict:
        """
        Stima l'orizzonte ottimale e restituisce un dict con:
          - recommended_days:  finestra consigliata (giorni)
          - recommended_label: etichetta leggibile (es. "3 mesi")
          - half_life_days:    half-life del regime stimato
          - autocorr_pvalue:   p-value Ljung-Box (basso = struttura temporale)
          - scores_df:         DataFrame dettaglio per ogni finestra candidata
          - confidence:        score composito della finestra scelta (0-1)
          - analysis:          dizionario con tutti i sotto-risultati
        """
        if len(self.returns) < max(self.candidates) + 63:
            logger.warning("Dati insufficienti per HorizonEstimator, uso default 6 mesi")
            return self._default_result()

        logger.info(f"HorizonEstimator: analisi {len(self.candidates)} finestre candidate")

        # 1. Stabilità pesi su rolling windows
        stability_scores = {}
        for w in self.candidates:
            stability_scores[w] = self._weight_stability(w)
            logger.debug(f"  Window {w}d — stability: {stability_scores[w]:.4f}")

        # 2. Sharpe OOS per ogni finestra
        sharpe_scores = {}
        for w in self.candidates:
            sharpe_scores[w] = self._oos_sharpe(w)
            logger.debug(f"  Window {w}d — OOS Sharpe: {sharpe_scores[w]:.3f}")

        # 3. Half-life del regime (AR-1)
        half_life = self._estimate_half_life()

        # 4. Test di autocorrelazione Ljung-Box
        autocorr_pval = self._ljung_box_pvalue()

        # 5. Score composito (normalizzato min-max)
        df = pd.DataFrame({
            "candidate_days": self.candidates,
            "stability_raw":  [stability_scores[w] for w in self.candidates],
            "oos_sharpe_raw": [sharpe_scores[w]    for w in self.candidates],
        }).set_index("candidate_days")

        # Normalizza stabilità: inversione (minore instabilità = punteggio alto)
        r_stab = df["stability_raw"].max() - df["stability_raw"].min()
        if r_stab > 0:
            df["stability_norm"] = 1 - (df["stability_raw"] - df["stability_raw"].min()) / r_stab
        else:
            df["stability_norm"] = 1.0

        # Normalizza Sharpe (più alto = meglio)
        r_sharpe = df["oos_sharpe_raw"].max() - df["oos_sharpe_raw"].min()
        if r_sharpe > 0:
            df["sharpe_norm"] = (df["oos_sharpe_raw"] - df["oos_sharpe_raw"].min()) / r_sharpe
        else:
            df["sharpe_norm"] = 0.5

        # Costo revisione: meno frequente = costo minore (finestre più lunghe favorevoli)
        df["cost_norm"] = pd.Series(
            np.linspace(0, 1, len(self.candidates)), index=self.candidates
        )

        # Score composito
        df["composite_score"] = (
            self.w_stability * df["stability_norm"] +
            self.w_sharpe    * df["sharpe_norm"]    +
            self.w_cost      * df["cost_norm"]
        )

        best_window = int(df["composite_score"].idxmax())
        confidence  = float(df.loc[best_window, "composite_score"])

        # Aggiusta in base all'autocorrelazione: se struttura temporale forte, accorcia
        adjusted_window = self._adjust_for_autocorr(best_window, autocorr_pval)

        # Aggiusta in base all'half-life
        final_window = self._adjust_for_halflife(adjusted_window, half_life)
        final_window = max(self.min_days, min(self.max_days, final_window))

        result = {
            "recommended_days":  final_window,
            "recommended_label": self._days_to_label(final_window),
            "best_raw_days":     best_window,
            "half_life_days":    round(half_life, 1),
            "autocorr_pvalue":   round(autocorr_pval, 4),
            "confidence":        round(confidence, 3),
            "scores_df":         df.reset_index(),
            "analysis": {
                "stability_scores": stability_scores,
                "sharpe_scores":    sharpe_scores,
                "half_life":        half_life,
                "autocorr_pval":    autocorr_pval,
                "interpretation":   self._interpret(final_window, half_life, autocorr_pval),
            }
        }
        logger.info(f"HorizonEstimator → {final_window} giorni ({self._days_to_label(final_window)}), "
                    f"confidence={confidence:.2f}, half_life={half_life:.0f}d")
        return result

    # ─── METODI ANALISI ──────────────────────────────────────────────────────

    def _weight_stability(self, window_days: int, step_days: int = 10) -> float:
        """
        Misura la variabilità dei pesi ottimali su finestre rolling.
        Output: std media dei pesi tra finestre (valore basso = stabile).
        """
        weights_list = []
        rets = self.returns

        for start in range(0, len(rets) - window_days, step_days):
            window = rets.iloc[start: start + window_days]
            if len(window) < max(30, window_days // 3):
                continue
            w = self._fast_optimize(window)
            if w is not None:
                weights_list.append(w)

        if len(weights_list) < 3:
            return 1.0  # Alta instabilità se non ci sono abbastanza dati

        arr = np.array(weights_list)
        return float(arr.std(axis=0).mean())

    def _oos_sharpe(self, window_days: int) -> float:
        """
        Calcola lo Sharpe ratio out-of-sample per la finestra data.
        Allena su window_days, valuta sui successivi window_days//2.
        Aggrega su più fold con step=window_days//4.
        """
        rets = self.returns
        test_days = max(21, window_days // 2)
        step      = max(10, window_days // 4)
        sharpes   = []

        for start in range(0, len(rets) - window_days - test_days, step):
            train = rets.iloc[start: start + window_days]
            test  = rets.iloc[start + window_days: start + window_days + test_days]

            if len(train) < 30 or len(test) < 10:
                continue

            w = self._fast_optimize(train)
            if w is None:
                continue

            port_rets = (test * w).sum(axis=1)
            ann_ret   = port_rets.mean() * 252
            ann_vol   = port_rets.std() * np.sqrt(252)

            if ann_vol > 1e-6:
                sharpes.append((ann_ret - RISK_FREE_RATE) / ann_vol)

        return float(np.median(sharpes)) if sharpes else 0.0

    def _estimate_half_life(self) -> float:
        """
        Stima il half-life (giorni) della stabilità dei pesi tramite AR(1).

        L'idea: calcola le differenze tra pesi ottimali consecutivi e
        adatta un AR(1) per stimare la velocità di mean-reversion.
        Half-life = -log(2) / log(|beta_AR1|)
        """
        step = 5
        weights_list = []
        rets = self.returns
        window = min(126, len(rets) // 3)  # 6 mesi o meno se dati scarsi

        for start in range(0, len(rets) - window, step):
            w = self._fast_optimize(rets.iloc[start: start + window])
            if w is not None:
                weights_list.append(w)

        if len(weights_list) < 6:
            return 63.0  # Default: 3 mesi

        arr = np.array(weights_list)
        # Norma delle differenze di pesi
        diffs = np.linalg.norm(np.diff(arr, axis=0), axis=1)

        if len(diffs) < 4:
            return 63.0

        # Regressione AR(1)
        y_lag = diffs[:-1]
        y_cur = diffs[1:]
        try:
            X = np.column_stack([np.ones_like(y_lag), y_lag])
            beta = np.linalg.lstsq(X, y_cur, rcond=None)[0]
            ar_coef = beta[1]

            if 0 < ar_coef < 1:
                hl = -np.log(2) / np.log(ar_coef)
                return max(21.0, min(252.0, hl * step))  # Converti passi → giorni
        except Exception:
            pass

        return 63.0  # Default sicuro

    def _ljung_box_pvalue(self, lags: int = 10) -> float:
        """
        Test di Ljung-Box sui rendimenti del portafoglio equally-weighted.
        p-value basso (<0.05) indica struttura temporale nei rendimenti.
        """
        try:
            port_rets = self.returns.mean(axis=1)
            n = len(port_rets)
            lags = min(lags, n // 5)

            # Ljung-Box manuale (per non richiedere statsmodels)
            acf_vals = [port_rets.autocorr(lag=k) for k in range(1, lags + 1)]
            acf_vals = [v if not np.isnan(v) else 0 for v in acf_vals]
            lb_stat  = n * (n + 2) * sum(
                v**2 / (n - k) for k, v in enumerate(acf_vals, 1)
            )

            from scipy.stats import chi2
            pval = 1 - chi2.cdf(lb_stat, df=lags)
            return float(pval)

        except Exception:
            return 0.5  # Valore neutro se errore

    # ─── OTTIMIZZAZIONE RAPIDA ───────────────────────────────────────────────

    def _fast_optimize(self, rets: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Ottimizzazione rapida dei pesi (max Sharpe semplificato).
        Restituisce None se fallisce.
        """
        try:
            n = rets.shape[1]
            if n == 0:
                return None

            mu  = rets.mean().values * 252
            cov = rets.cov().values * 252
            w0  = np.ones(n) / n

            def neg_sharpe(w):
                ret = w @ mu
                vol = np.sqrt(w @ cov @ w)
                if vol < 1e-8:
                    return 1e6
                return -(ret - RISK_FREE_RATE) / vol

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    neg_sharpe, w0,
                    method="SLSQP",
                    bounds=[(MIN_WEIGHT, MAX_WEIGHT)] * n,
                    constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                    options={"maxiter": 200, "ftol": 1e-7}
                )

            if res.success:
                w = np.clip(res.x, MIN_WEIGHT, MAX_WEIGHT)
                return w / w.sum()

            # Fallback: pesi equal-weight
            return w0

        except Exception:
            return None

    # ─── AGGIUSTAMENTI FINALI ────────────────────────────────────────────────

    def _adjust_for_autocorr(self, window: int, pval: float) -> int:
        """
        Se c'è autocorrelazione significativa nei rendimenti (pval < 0.05),
        suggerisce un orizzonte più breve (mercato più prevedibile a breve).
        """
        if pval < 0.05:
            # Struttura temporale forte: privilegia finestra più corta del 30%
            shorter = int(window * 0.70)
            shorter = max(self.min_days, shorter)
            logger.debug(f"Autocorr significativa (p={pval:.3f}): accorcia {window}→{shorter}d")
            return shorter
        return window

    def _adjust_for_halflife(self, window: int, half_life: float) -> int:
        """
        Aggusta la finestra in base all'half-life stimato.
        La finestra ottimale è circa 2-3 volte l'half-life.
        """
        optimal_from_hl = int(half_life * 2.0)
        optimal_from_hl = max(self.min_days, min(self.max_days, optimal_from_hl))

        # Media pesata (60% stima statistica, 40% half-life)
        blended = int(0.60 * window + 0.40 * optimal_from_hl)
        # Arrotonda alla finestra candidata più vicina
        closest = min(self.candidates, key=lambda c: abs(c - blended))
        return closest

    # ─── UTILITY ─────────────────────────────────────────────────────────────

    def _days_to_label(self, days: int) -> str:
        if days <= 25:
            return "1 mese"
        elif days <= 55:
            return "2 mesi"
        elif days <= 80:
            return "3 mesi"
        elif days <= 140:
            return "6 mesi"
        elif days <= 210:
            return "9 mesi"
        return "12 mesi"

    def _interpret(self, days: int, half_life: float, pval: float) -> str:
        parts = []
        parts.append(f"Orizzonte consigliato: {days} giorni trading "
                     f"({self._days_to_label(days)})")
        parts.append(f"Half-life stabilità pesi: {half_life:.0f} giorni")
        if pval < 0.05:
            parts.append(f"Test Ljung-Box: autocorrelazione significativa (p={pval:.3f}) → "
                         "mercato con struttura a breve termine, revisioni più frequenti")
        else:
            parts.append(f"Test Ljung-Box: rendimenti sostanzialmente indipendenti (p={pval:.3f}) → "
                         "revisioni meno frequenti sufficienti")
        return " | ".join(parts)

    def _default_result(self) -> dict:
        return {
            "recommended_days":  126,
            "recommended_label": "6 mesi (default)",
            "best_raw_days":     126,
            "half_life_days":    63.0,
            "autocorr_pvalue":   0.5,
            "confidence":        0.5,
            "scores_df":         pd.DataFrame(),
            "analysis":          {"interpretation": "Dati insufficienti, usato default 6 mesi"},
        }
