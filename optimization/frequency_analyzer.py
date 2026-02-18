"""
╔══════════════════════════════════════════════════════════════╗
║         FREQUENCY SCENARIO ANALYZER — v7                     ║
║    Confronto scenari per frequenza di revisione portafoglio  ║
╚══════════════════════════════════════════════════════════════╝

Simula la performance del portafoglio con diverse frequenze di
revisione e individua quella ottimale per risk-adjusted return.

Frequenze simulate:
  - Mensile      (21 giorni trading)
  - Trimestrale  (63 giorni)
  - Semestrale   (126 giorni)
  - Annuale      (252 giorni)
  - Dinamica     (revisione trigggerata da segnali: vol o drawdown)

Metriche calcolate per ciascuno scenario:
  CAGR, Volatilità, Max Drawdown, Sharpe, Net Sharpe (dopo costi),
  Turnover annuo, Costi totali %, Numero revisioni/anno

Utilizzo:
    analyzer = FrequencyScenarioAnalyzer(price_data, capital=10000)
    df       = analyzer.run_all_scenarios()
    best     = analyzer.suggest_optimal(df)
    analyzer.print_report(df, best)
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger("portfolio_optimizer")

RISK_FREE_RATE = 0.035
RISK_AVERSION  = 2.0
MIN_WEIGHT     = 0.03
MAX_WEIGHT     = 0.20


class FrequencyScenarioAnalyzer:
    """
    Confronta strategie di revisione a diverse frequenze.

    Args:
        price_data:       DataFrame prezzi (index=date, cols=ticker)
        transaction_cost: Costo percentuale per operazione (default 0.1%)
        capital:          Capitale simulato (per calcolo costi assoluti)
        min_train_days:   Minimo giorni di dati per prima ottimizzazione
    """

    FREQUENCIES = {
        "monthly":    {"days": 21,   "label": "Mensile"},
        "quarterly":  {"days": 63,   "label": "Trimestrale"},
        "semiannual": {"days": 126,  "label": "Semestrale"},
        "annual":     {"days": 252,  "label": "Annuale"},
        "dynamic":    {"days": None, "label": "Dinamica"},
    }

    # Parametri per la modalità dinamica
    DYN_VOL_THRESHOLD  = 0.25    # Revisione se vol rolling 20gg > 25% annualizzata
    DYN_DD_THRESHOLD   = -0.08   # Revisione se drawdown corrente < -8%
    DYN_MIN_GAP_DAYS   = 21      # Almeno 21gg tra una revisione e l'altra

    def __init__(
        self,
        price_data:       pd.DataFrame,
        transaction_cost: float = 0.001,
        capital:          float = 10_000.0,
        min_train_days:   int   = 126,
    ):
        self.prices    = price_data.dropna(axis=1, how="all")
        self.returns   = self.prices.pct_change().dropna()
        self.tc        = transaction_cost   # 0.001 = 0.1%
        self.capital   = capital
        self.min_train = min_train_days
        self.tickers   = list(self.prices.columns)
        self.n         = len(self.tickers)

        if len(self.returns) < self.min_train:
            raise ValueError(
                f"Dati insufficienti: {len(self.returns)} righe, "
                f"minimo {self.min_train}"
            )

    # ─── METODO PRINCIPALE ───────────────────────────────────────────────────

    def run_all_scenarios(self) -> pd.DataFrame:
        """
        Esegue tutti gli scenari e restituisce un DataFrame comparativo,
        ordinato per Net Sharpe decrescente.
        """
        results = []
        for key, meta in self.FREQUENCIES.items():
            logger.info(f"FrequencyAnalyzer: simulazione '{meta['label']}'...")
            try:
                if key == "dynamic":
                    metrics = self._run_dynamic()
                else:
                    metrics = self._run_fixed(meta["days"])
                metrics["frequency"] = meta["label"]
                metrics["freq_key"]  = key
                results.append(metrics)
            except Exception as e:
                logger.warning(f"Scenario {key} fallito: {e}")
                continue

        if not results:
            logger.error("Nessuno scenario completato con successo")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Score ottimale composito (per ranking)
        df["optimal_score"] = self._compute_optimal_scores(df)
        df = df.sort_values("optimal_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        return df

    def suggest_optimal(self, df: pd.DataFrame) -> dict:
        """
        Restituisce la frequenza consigliata con motivazione dettagliata.
        """
        if df.empty:
            return {"frequency": "Semestrale", "rationale": "Dati insufficienti"}

        best = df.iloc[0]
        second = df.iloc[1] if len(df) > 1 else None

        rationale_parts = [
            f"La frequenza '{best['frequency']}' ottiene il punteggio complessivo più alto.",
            f"Net Sharpe: {best['net_sharpe']:.2f}  |  "
            f"Turnover annuo: {best['turnover_annual']:.0%}  |  "
            f"Costi annui: {best['total_costs_pct']:.2%}",
        ]
        if second is not None:
            rationale_parts.append(
                f"Alternativa: '{second['frequency']}' (Net Sharpe: {second['net_sharpe']:.2f})"
            )

        return {
            "frequency":       best["frequency"],
            "freq_key":        best["freq_key"],
            "days":            self.FREQUENCIES.get(best["freq_key"], {}).get("days"),
            "net_sharpe":      round(float(best["net_sharpe"]), 3),
            "cagr":            round(float(best["cagr"]), 4),
            "volatility":      round(float(best["volatility"]), 4),
            "max_drawdown":    round(float(best["max_drawdown"]), 4),
            "turnover_annual": round(float(best["turnover_annual"]), 4),
            "total_costs_pct": round(float(best["total_costs_pct"]), 4),
            "n_revisions":     int(best["n_revisions"]),
            "rationale":       " ".join(rationale_parts),
        }

    def print_report(self, df: pd.DataFrame, best: dict):
        """Stampa il report comparativo su console."""
        print("\n" + "═"*72)
        print("  📅  ANALISI SCENARI — FREQUENZA DI REVISIONE")
        print("═"*72)
        print(f"  {'Frequenza':<14} {'CAGR':>7} {'Vol':>7} {'MDD':>8} "
              f"{'Sharpe':>7} {'NetShr':>7} {'Turn':>7} {'Costi%':>7} {'Score':>7}")
        print("  " + "─"*70)

        for _, row in df.iterrows():
            marker = "◄" if row["frequency"] == best["frequency"] else " "
            print(f"  {row['frequency']:<14} "
                  f"{row['cagr']:>6.1%} "
                  f"{row['volatility']:>6.1%} "
                  f"{row['max_drawdown']:>7.1%} "
                  f"{row['sharpe']:>7.2f} "
                  f"{row['net_sharpe']:>7.2f} "
                  f"{row['turnover_annual']:>6.0%} "
                  f"{row['total_costs_pct']:>6.2%} "
                  f"{row['optimal_score']:>7.3f} {marker}")

        print("═"*72)
        print(f"\n  🏆 FREQUENZA OTTIMALE: {best['frequency'].upper()}")
        print(f"  {best['rationale']}")
        print("═"*72 + "\n")

    # ─── SIMULAZIONE FREQUENZA FISSA ─────────────────────────────────────────

    def _run_fixed(self, rebalance_days: int) -> dict:
        """
        Simula la strategia con revisione a frequenza fissa.
        """
        rets       = self.returns
        n_rows     = len(rets)
        port_rets  = []
        total_turn = 0.0
        n_revs     = 0
        prev_w     = None

        i = 0
        while i < n_rows:
            # Prima revisione: usa i dati disponibili fino a i (min train)
            if i < self.min_train:
                i += 1
                continue

            # Ottimizza su dati storici fino a i
            train = rets.iloc[:i]
            new_w = self._optimize(train)

            # Calcola turnover
            if prev_w is not None:
                turn = float(np.sum(np.abs(new_w - prev_w)) / 2)
                total_turn += turn
            prev_w = new_w
            n_revs += 1

            # Applica nuovi pesi fino alla prossima revisione
            end = min(i + rebalance_days, n_rows)
            period_rets = rets.iloc[i:end]
            p_rets      = (period_rets * new_w).sum(axis=1)
            port_rets.extend(p_rets.tolist())

            i = end

        if not port_rets:
            return self._empty_metrics()

        # Anni effettivi simulati
        years_sim = len(port_rets) / 252

        metrics = self._compute_metrics(
            port_rets, total_turn, n_revs, years_sim
        )
        return metrics

    # ─── SIMULAZIONE DINAMICA ─────────────────────────────────────────────────

    def _run_dynamic(self) -> dict:
        """
        Simula la strategia con frequenza di revisione dinamica.

        Trigger di revisione (OR):
          - Volatilità rolling 20gg > DYN_VOL_THRESHOLD annualizzata
          - Drawdown corrente dalla cresta < DYN_DD_THRESHOLD
          - Almeno DYN_MIN_GAP_DAYS dall'ultima revisione
        """
        rets       = self.returns
        n_rows     = len(rets)
        port_rets  = []
        total_turn = 0.0
        n_revs     = 0
        prev_w     = None

        # Inizializza pesi uniformi
        curr_w     = np.ones(self.n) / self.n
        last_rev   = self.min_train
        cum_val    = 1.0  # Valore cumulativo normalizzato
        peak_val   = 1.0

        for i in range(self.min_train, n_rows):
            day_ret    = rets.iloc[i]
            day_portrt = float((day_ret * curr_w).sum())
            cum_val   *= (1 + day_portrt)
            peak_val   = max(peak_val, cum_val)

            # Metriche di trigger
            days_since_rev = i - last_rev

            # Volatilità rolling 20gg
            if i >= 20:
                win20 = rets.iloc[i-20:i]
                roll_vol = float((win20 * curr_w).sum(axis=1).std() * np.sqrt(252))
            else:
                roll_vol = 0.0

            # Drawdown corrente
            drawdown = (cum_val - peak_val) / peak_val if peak_val > 0 else 0.0

            # Trigger revisione
            vol_trigger = roll_vol > self.DYN_VOL_THRESHOLD
            dd_trigger  = drawdown < self.DYN_DD_THRESHOLD
            should_rev  = (vol_trigger or dd_trigger) and \
                          days_since_rev >= self.DYN_MIN_GAP_DAYS

            if should_rev:
                train  = rets.iloc[:i]
                new_w  = self._optimize(train)

                if prev_w is not None:
                    turn = float(np.sum(np.abs(new_w - curr_w)) / 2)
                    total_turn += turn

                curr_w   = new_w
                prev_w   = new_w
                last_rev = i
                n_revs  += 1

            port_rets.append(day_portrt)

        if not port_rets:
            return self._empty_metrics()

        years_sim = len(port_rets) / 252
        return self._compute_metrics(port_rets, total_turn, n_revs, years_sim)

    # ─── METRICHE ────────────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        port_rets: list,
        total_turnover: float,
        n_revisions: int,
        years: float,
    ) -> dict:
        """Calcola tutte le metriche per uno scenario."""
        rets_arr = np.array(port_rets)
        cum      = np.cumprod(1 + rets_arr)

        # CAGR
        if years > 0 and cum[-1] > 0:
            cagr = float(cum[-1] ** (1 / years) - 1)
        else:
            cagr = 0.0

        # Volatilità annualizzata
        vol = float(rets_arr.std() * np.sqrt(252))

        # Max Drawdown
        peak = np.maximum.accumulate(cum)
        dd   = (cum - peak) / np.where(peak > 0, peak, 1)
        mdd  = float(dd.min())

        # Sharpe (lordo)
        sharpe = (cagr - RISK_FREE_RATE) / vol if vol > 1e-8 else 0.0

        # Turnover annuo e costi
        turn_annual = total_turnover / years if years > 0 else 0.0
        # Costi totali: ogni unità di turnover costa tc% (su acquisti E vendite)
        costs_pct   = turn_annual * self.tc * 2  # moltiplicato 2 per round-trip
        cagr_net    = cagr - costs_pct
        net_sharpe  = (cagr_net - RISK_FREE_RATE) / vol if vol > 1e-8 else 0.0

        # Numero revisioni per anno
        revs_per_year = n_revisions / years if years > 0 else 0.0

        # Calmar ratio
        calmar = cagr / abs(mdd) if mdd < 0 else 0.0

        return {
            "cagr":             cagr,
            "cagr_net":         cagr_net,
            "volatility":       vol,
            "max_drawdown":     mdd,
            "sharpe":           sharpe,
            "net_sharpe":       net_sharpe,
            "calmar":           calmar,
            "turnover_annual":  turn_annual,
            "total_costs_pct":  costs_pct,
            "n_revisions":      n_revisions,
            "revisions_per_year": revs_per_year,
            "years_simulated":  years,
        }

    def _compute_optimal_scores(self, df: pd.DataFrame) -> pd.Series:
        """
        Score composito per ranking (più alto = migliore):
          60% Net Sharpe normalizzato
          20% 1 - (turnover normalizzato)   [meno rotazione = meglio]
          20% 1 - (costi normalizzati)
        """
        def norm(series):
            r = series.max() - series.min()
            return (series - series.min()) / r if r > 1e-8 else pd.Series(0.5, index=series.index)

        score = (
            0.60 * norm(df["net_sharpe"]) +
            0.20 * (1 - norm(df["turnover_annual"])) +
            0.20 * (1 - norm(df["total_costs_pct"]))
        )
        return score

    # ─── OTTIMIZZAZIONE INTERNA ───────────────────────────────────────────────

    def _optimize(self, rets: pd.DataFrame) -> np.ndarray:
        """Max-Sharpe semplificato con vincoli."""
        n   = rets.shape[1]
        mu  = rets.mean().values * 252
        cov = rets.cov().values * 252
        w0  = np.ones(n) / n

        def neg_sharpe(w):
            r   = w @ mu
            vol = np.sqrt(w @ cov @ w)
            if vol < 1e-8:
                return 1e6
            return -(r - RISK_FREE_RATE) / vol

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                neg_sharpe, w0,
                method="SLSQP",
                bounds=[(MIN_WEIGHT, MAX_WEIGHT)] * n,
                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                options={"maxiter": 300, "ftol": 1e-8}
            )

        if res.success:
            w = np.clip(res.x, MIN_WEIGHT, MAX_WEIGHT)
            return w / w.sum()

        # Fallback: equal weight
        return w0

    def _empty_metrics(self) -> dict:
        return {
            "cagr": 0, "cagr_net": 0, "volatility": 0,
            "max_drawdown": 0, "sharpe": 0, "net_sharpe": 0, "calmar": 0,
            "turnover_annual": 0, "total_costs_pct": 0,
            "n_revisions": 0, "revisions_per_year": 0, "years_simulated": 0,
        }
