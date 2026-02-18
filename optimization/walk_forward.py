"""
╔══════════════════════════════════════════════════════════════╗
║         WALK-FORWARD VALIDATOR — v7                          ║
║   Backtesting iterativo rolling con metriche di robustezza   ║
╚══════════════════════════════════════════════════════════════╝

Implementa il backtesting walk-forward (out-of-sample rolling):
  - La strategia viene STIMATA su TRAIN set
  - VALUTATA su TEST set immediatamente successivo
  - Il processo è ripetuto spostando la finestra nel tempo
  - ZERO lookahead bias: i dati di test non vengono mai usati in training

Modalità disponibili:
  - Fixed window:    TRAIN di dimensione fissa, avanza di step_months
  - Expanding:       TRAIN cresce nel tempo (aggiunge dati, non scarta i vecchi)
  - Multi-horizon:   Testa automaticamente training da 1, 2, 3, 5, 10 anni

Output:
  - Metriche per-fold (CAGR, Sharpe, MaxDD, ...)
  - Metriche aggregate di robustezza
  - Consistency score (0-1)
  - Sensitivity analysis
  - Stability dei pesi tra fold

Utilizzo:
    wf = WalkForwardValidator(price_data, train_years=3, test_years=1)
    result = wf.run()
    wf.print_report(result)

    # Modalità expanding
    result_exp = wf.run(expanding=True)

    # Multi-horizon
    summary = wf.run_multi_horizon()
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger("portfolio_optimizer")

RISK_FREE_RATE = 0.035
RISK_AVERSION  = 2.0
MIN_WEIGHT     = 0.03
MAX_WEIGHT     = 0.20


# ─── DATACLASS RISULTATO ─────────────────────────────────────────────────────

@dataclass
class FoldResult:
    """Risultato di un singolo fold walk-forward."""
    fold_id:          int
    train_start:      str
    train_end:        str
    test_start:       str
    test_end:         str
    train_years:      float
    test_years:       float
    weights:          Dict[str, float]   # ticker → peso ottimale
    # Metriche out-of-sample
    cagr:             float
    volatility:       float
    sharpe_ratio:     float
    max_drawdown:     float
    calmar_ratio:     float
    total_return:     float
    # Confronto con benchmark equally-weighted
    bench_cagr:       float
    bench_sharpe:     float
    alpha:            float   # cagr - bench_cagr
    # Diagnostica
    n_assets:         int
    weight_hhi:       float   # Herfindahl Index (concentrazione)


# ─── WALK-FORWARD VALIDATOR ──────────────────────────────────────────────────

class WalkForwardValidator:
    """
    Validatore walk-forward per portafogli azionari.

    Args:
        price_data:    DataFrame prezzi (index=date, cols=ticker)
        train_years:   Anni di dati usati per training (default 3)
        test_years:    Anni di test OOS (default 1)
        step_months:   Avanzamento finestra in mesi (default 6)
        min_assets:    Minimo titoli in portafoglio
        max_assets:    Massimo titoli in portafoglio
    """

    def __init__(
        self,
        price_data:  pd.DataFrame,
        train_years: int   = 3,
        test_years:  int   = 1,
        step_months: int   = 6,
        min_assets:  int   = 5,
        max_assets:  int   = 20,
    ):
        self.prices      = price_data.dropna(axis=1, how="all")
        self.returns     = self.prices.pct_change().dropna()
        self.tickers     = list(self.prices.columns)
        self.n           = len(self.tickers)
        self.train_days  = int(train_years * 252)
        self.test_days   = int(test_years  * 252)
        self.step_days   = int(step_months * 21)
        self.min_assets  = min_assets
        self.max_assets  = max_assets

    # ─── METODO PRINCIPALE ───────────────────────────────────────────────────

    def run(self, expanding: bool = False) -> dict:
        """
        Esegue il walk-forward completo.

        Args:
            expanding: Se True, usa expanding window (TRAIN cresce nel tempo).
                       Se False (default), usa fixed window.

        Returns:
            Dict con: folds, aggregate_metrics, robustness_metrics,
                      consistency_score, sensitivity, weights_stability
        """
        n_rows = len(self.returns)
        min_needed = self.train_days + self.test_days
        if n_rows < min_needed:
            logger.warning(f"Dati insufficienti per walk-forward: "
                           f"{n_rows} righe, minimo {min_needed}")
            return self._empty_result()

        folds: List[FoldResult] = []
        start = 0
        fold_id = 1

        logger.info(f"WalkForward: modalità={'expanding' if expanding else 'fixed'}, "
                    f"train={self.train_days}d, test={self.test_days}d, step={self.step_days}d")

        while True:
            if expanding:
                train_start = 0
                train_end   = start + self.train_days
            else:
                train_start = start
                train_end   = start + self.train_days

            test_start = train_end
            test_end   = test_start + self.test_days

            if test_end > n_rows:
                break

            train_rets = self.returns.iloc[train_start:train_end]
            test_rets  = self.returns.iloc[test_start:test_end]

            logger.debug(f"Fold {fold_id}: train [{train_start}:{train_end}], "
                         f"test [{test_start}:{test_end}]")

            # 1. Fit su TRAIN (nessun accesso ai dati TEST)
            weights = self._fit(train_rets)

            # 2. Valuta su TEST (OOS puro)
            fold = self._evaluate(
                fold_id, weights, train_rets, test_rets
            )
            folds.append(fold)

            start   += self.step_days
            fold_id += 1

        if not folds:
            return self._empty_result()

        logger.info(f"WalkForward completato: {len(folds)} fold analizzati")
        return self._aggregate(folds)

    def run_multi_horizon(self) -> dict:
        """
        Esegue il walk-forward per ognuno degli orizzonti storici:
        1, 2, 3, 5, 10 anni di training, test sempre di 1 anno.

        Restituisce un dict {train_years: result} con confronto tra orizzonti.
        """
        horizons = [1, 2, 3, 5, 10]
        results = {}

        for hy in horizons:
            required = hy * 252 + self.test_days
            if len(self.returns) < required:
                logger.info(f"Training {hy}a: dati insufficienti, salto")
                continue

            logger.info(f"WalkForward multi-horizon: training {hy} anni...")
            prev_train = self.train_days
            self.train_days = int(hy * 252)
            result = self.run()
            self.train_days = prev_train
            result["train_years"] = hy
            results[hy] = result

        return self._compare_horizons(results)

    def print_report(self, result: dict, verbose: bool = True):
        """Stampa il report del walk-forward su console."""
        if result.get("n_folds", 0) == 0:
            print("  ⚠️  Nessun fold completato (dati insufficienti)")
            return

        print("\n" + "═"*70)
        print("  🔬  WALK-FORWARD VALIDATION — RISULTATI")
        print("═"*70)

        agg = result["aggregate"]
        rob = result["robustness"]

        print(f"\n  📊 METRICHE AGGREGATE ({result['n_folds']} fold)")
        print(f"  {'Metrica':<28} {'Media':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print("  " + "─"*60)
        for col, label in [
            ("cagr",         "CAGR OOS"),
            ("volatility",   "Volatilità"),
            ("sharpe_ratio", "Sharpe Ratio"),
            ("max_drawdown", "Max Drawdown"),
            ("alpha",        "Alpha vs EW"),
        ]:
            vals = [f.cagr if col=="cagr" else
                    f.volatility if col=="volatility" else
                    f.sharpe_ratio if col=="sharpe_ratio" else
                    f.max_drawdown if col=="max_drawdown" else
                    f.alpha
                    for f in result["folds"]]
            arr = np.array(vals)
            fmt = ".1%" if col in ("cagr","volatility","max_drawdown","alpha") else ".3f"
            print(f"  {label:<28} {arr.mean():>8{fmt}} {arr.std():>8{fmt}} "
                  f"{arr.min():>8{fmt}} {arr.max():>8{fmt}}")

        print(f"\n  🛡️  METRICHE DI ROBUSTEZZA")
        print(f"  Consistency Score:    {rob['consistency_score']:.3f}  "
              f"(0=scarso, 1=eccellente)")
        print(f"  % Fold positivi:      {rob['pct_positive_folds']:.0%}")
        print(f"  Stabilità pesi:       {rob['weight_stability']:.4f}  "
              f"(minore = più stabile)")
        print(f"  Sharpe trend tempo:   {rob['sharpe_trend']:+.3f}  "
              f"({'migliora' if rob['sharpe_trend'] > 0 else 'peggiora'} col tempo)")

        if verbose and result["folds"]:
            print(f"\n  📋 DETTAGLIO FOLD")
            print(f"  {'Fold':<5} {'Train':<22} {'Test fine':<12} "
                  f"{'CAGR':>7} {'Sharpe':>7} {'MDD':>8} {'Alpha':>7}")
            print("  " + "─"*70)
            for f in result["folds"]:
                print(f"  {f.fold_id:<5} {f.train_start[:10]}→{f.train_end[:10]}  "
                      f"{f.test_end[:10]}  "
                      f"{f.cagr:>6.1%} {f.sharpe_ratio:>7.2f} "
                      f"{f.max_drawdown:>7.1%} {f.alpha:>+6.1%}")

        print("═"*70 + "\n")

    # ─── FITTING E VALUTAZIONE ────────────────────────────────────────────────

    def _fit(self, train_rets: pd.DataFrame) -> Dict[str, float]:
        """
        Stima i pesi ottimali usando SOLO i dati di training.
        Restituisce un dict {ticker: weight}.
        """
        n   = train_rets.shape[1]
        mu  = train_rets.mean().values * 252
        cov = train_rets.cov().values * 252
        w0  = np.ones(n) / n

        def objective(w):
            ret = w @ mu
            vol = np.sqrt(w @ cov @ w)
            if vol < 1e-8:
                return 1e6
            # Utility: rendimento - avversione * var - penalità vol individuale
            vols_ind = np.sqrt(np.diag(cov))
            penalty  = np.dot(w, vols_ind) * RISK_AVERSION * 0.3
            return -(ret - RISK_FREE_RATE) / vol - 0.1 * (-RISK_AVERSION * vol - penalty)

        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(w @ cov @ w)
            return -(ret - RISK_FREE_RATE) / vol if vol > 1e-8 else 1e6

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                neg_sharpe, w0,
                method="SLSQP",
                bounds=[(MIN_WEIGHT, MAX_WEIGHT)] * n,
                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                options={"maxiter": 500, "ftol": 1e-9}
            )

        if res.success:
            w = np.clip(res.x, MIN_WEIGHT, MAX_WEIGHT)
            w = w / w.sum()
        else:
            w = w0

        return {t: float(wi) for t, wi in zip(train_rets.columns, w)}

    def _evaluate(
        self,
        fold_id:    int,
        weights:    Dict[str, float],
        train_rets: pd.DataFrame,
        test_rets:  pd.DataFrame,
    ) -> FoldResult:
        """Valuta la strategia sul test set OOS."""
        tickers_w = [t for t in weights if t in test_rets.columns]
        if not tickers_w:
            return self._empty_fold(fold_id, train_rets, test_rets)

        w_arr   = np.array([weights[t] for t in tickers_w])
        w_arr  /= w_arr.sum()
        rets_m  = test_rets[tickers_w].values

        # Rendimenti portafoglio
        port_rets = rets_m @ w_arr

        # Benchmark equally-weighted
        bench_rets = test_rets[tickers_w].mean(axis=1).values

        # Calcola metriche OOS
        def metrics_from_rets(r):
            n_days   = len(r)
            years    = n_days / 252
            cum      = np.cumprod(1 + r)
            tot_ret  = float(cum[-1] - 1) if len(cum) > 0 else 0.0
            cagr     = float((1 + tot_ret) ** (1 / years) - 1) if years > 0 else 0.0
            vol      = float(r.std() * np.sqrt(252))
            sharpe   = (cagr - RISK_FREE_RATE) / vol if vol > 1e-8 else 0.0
            peak     = np.maximum.accumulate(cum)
            dd       = (cum - peak) / np.where(peak > 0, peak, 1)
            mdd      = float(dd.min()) if len(dd) > 0 else 0.0
            calmar   = cagr / abs(mdd) if mdd < -1e-6 else 0.0
            return tot_ret, cagr, vol, sharpe, mdd, calmar

        t_ret, t_cagr, t_vol, t_shr, t_mdd, t_cal = metrics_from_rets(port_rets)
        _, b_cagr, _, b_shr, _, _ = metrics_from_rets(bench_rets)

        # Herfindahl Index (concentrazione pesi)
        hhi = float(np.sum(w_arr ** 2))

        train_start_date = str(train_rets.index[0])[:10]
        train_end_date   = str(train_rets.index[-1])[:10]
        test_start_date  = str(test_rets.index[0])[:10]
        test_end_date    = str(test_rets.index[-1])[:10]

        return FoldResult(
            fold_id      = fold_id,
            train_start  = train_start_date,
            train_end    = train_end_date,
            test_start   = test_start_date,
            test_end     = test_end_date,
            train_years  = len(train_rets) / 252,
            test_years   = len(test_rets)  / 252,
            weights      = weights,
            cagr         = t_cagr,
            volatility   = t_vol,
            sharpe_ratio = t_shr,
            max_drawdown = t_mdd,
            calmar_ratio = t_cal,
            total_return = t_ret,
            bench_cagr   = b_cagr,
            bench_sharpe = b_shr,
            alpha        = t_cagr - b_cagr,
            n_assets     = len(tickers_w),
            weight_hhi   = hhi,
        )

    # ─── AGGREGAZIONE ────────────────────────────────────────────────────────

    def _aggregate(self, folds: List[FoldResult]) -> dict:
        """Aggrega i fold in metriche di robustezza."""
        n = len(folds)

        cagrs   = np.array([f.cagr        for f in folds])
        sharpes = np.array([f.sharpe_ratio for f in folds])
        mdds    = np.array([f.max_drawdown for f in folds])
        alphas  = np.array([f.alpha        for f in folds])
        vols    = np.array([f.volatility   for f in folds])

        # Stabilità pesi tra fold
        all_tickers = sorted(set().union(*[set(f.weights.keys()) for f in folds]))
        weight_matrix = np.array([
            [f.weights.get(t, 0) for t in all_tickers]
            for f in folds
        ])
        weight_stability = float(weight_matrix.std(axis=0).mean())

        # Trend Sharpe nel tempo (correlazione Sharpe vs indice fold)
        if n > 2:
            fold_ids     = np.arange(n)
            sharpe_trend = float(np.corrcoef(fold_ids, sharpes)[0, 1])
        else:
            sharpe_trend = 0.0

        # Consistency score: combina % fold positivi e variabilità Sharpe
        pct_pos  = float((cagrs > 0).mean())
        cv_sharpe = sharpes.std() / max(abs(sharpes.mean()), 0.01)
        consistency = pct_pos * 0.50 + max(0, 1 - cv_sharpe) * 0.50
        consistency = float(np.clip(consistency, 0, 1))

        # Information ratio (alpha medio / std alpha)
        ir = float(alphas.mean() / alphas.std()) if alphas.std() > 1e-6 else 0.0

        robustness = {
            "consistency_score":   round(consistency, 3),
            "pct_positive_folds":  round(pct_pos, 3),
            "weight_stability":    round(weight_stability, 5),
            "sharpe_trend":        round(sharpe_trend, 3),
            "information_ratio":   round(ir, 3),
            "mean_alpha":          round(float(alphas.mean()), 4),
            "std_alpha":           round(float(alphas.std()),  4),
            "mean_sharpe":         round(float(sharpes.mean()), 3),
            "std_sharpe":          round(float(sharpes.std()),  3),
            "cv_sharpe":           round(float(cv_sharpe), 3),
        }

        aggregate = {
            "mean_cagr":       round(float(cagrs.mean()), 4),
            "std_cagr":        round(float(cagrs.std()),  4),
            "mean_sharpe":     round(float(sharpes.mean()), 3),
            "std_sharpe":      round(float(sharpes.std()),  3),
            "mean_max_dd":     round(float(mdds.mean()), 4),
            "mean_vol":        round(float(vols.mean()), 4),
            "mean_alpha":      round(float(alphas.mean()), 4),
            "min_cagr":        round(float(cagrs.min()), 4),
            "max_cagr":        round(float(cagrs.max()), 4),
        }

        return {
            "n_folds":     n,
            "folds":       folds,
            "aggregate":   aggregate,
            "robustness":  robustness,
            "consistency_score": consistency,
            "tickers_used": all_tickers,
        }

    def _compare_horizons(self, results: dict) -> dict:
        """Confronta i risultati per diverse finestre di training."""
        rows = []
        for hy, res in results.items():
            if res.get("n_folds", 0) == 0:
                continue
            rob = res["robustness"]
            agg = res["aggregate"]
            rows.append({
                "train_years":       hy,
                "n_folds":           res["n_folds"],
                "mean_cagr":         agg["mean_cagr"],
                "mean_sharpe":       agg["mean_sharpe"],
                "mean_max_dd":       agg["mean_max_dd"],
                "consistency_score": rob["consistency_score"],
                "pct_positive":      rob["pct_positive_folds"],
                "weight_stability":  rob["weight_stability"],
                "sharpe_trend":      rob["sharpe_trend"],
            })
        df = pd.DataFrame(rows)

        # Scegli l'orizzonte migliore: max consistency × mean_sharpe
        if not df.empty:
            df["composite"] = df["consistency_score"] * 0.5 + \
                              df["mean_sharpe"].clip(lower=0) / df["mean_sharpe"].clip(lower=0).max().clip(lower=0.01) * 0.5
            best_hy = int(df.loc[df["composite"].idxmax(), "train_years"])
        else:
            best_hy = 3

        return {
            "comparison_df":       df,
            "best_train_years":    best_hy,
            "per_horizon_results": results,
        }

    # ─── UTILITY ─────────────────────────────────────────────────────────────

    def _empty_fold(self, fold_id, train_rets, test_rets) -> FoldResult:
        return FoldResult(
            fold_id=fold_id,
            train_start=str(train_rets.index[0])[:10] if len(train_rets) > 0 else "",
            train_end=str(train_rets.index[-1])[:10] if len(train_rets) > 0 else "",
            test_start=str(test_rets.index[0])[:10] if len(test_rets) > 0 else "",
            test_end=str(test_rets.index[-1])[:10] if len(test_rets) > 0 else "",
            train_years=0, test_years=0, weights={},
            cagr=0, volatility=0, sharpe_ratio=0, max_drawdown=0,
            calmar_ratio=0, total_return=0, bench_cagr=0, bench_sharpe=0,
            alpha=0, n_assets=0, weight_hhi=0
        )

    def _empty_result(self) -> dict:
        return {
            "n_folds": 0, "folds": [],
            "aggregate": {}, "robustness": {},
            "consistency_score": 0,
            "tickers_used": [],
        }
