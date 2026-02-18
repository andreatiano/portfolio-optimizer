"""
╔══════════════════════════════════════════════════════════════╗
║      PORTFOLIO OPTIMIZER v7 — Main Entry Point               ║
╚══════════════════════════════════════════════════════════════╝

MODALITÀ DI UTILIZZO
─────────────────────────────────────────────────────────────────

  ① PRIMA VOLTA — crea il portafoglio da zero:
      python main.py --capital 10000

      → scarica dati, ottimizza, genera report HTML
      → salva automaticamente  portfolio_state.json
      → ti dice quando fare il prossimo rebalance

  ② AGGIORNA PREZZI — prima di un rebalance (veloce, ~10 sec):
      python main.py --update-prices

      → scarica solo i prezzi correnti
      → mostra P&L non realizzato per ogni titolo
      → aggiorna portfolio_state.json  (non ricalcola niente)

  ③ REBALANCE — al momento della revisione:
      python main.py --rebalance --capital 10500

      → carica portfolio_state.json  (stato precedente)
      → ricalcola l'ottimizzazione con i dati aggiornati
      → mostra gli ordini SELL/BUY da eseguire
      → aggiorna portfolio_state.json  con il nuovo stato
      → genera report HTML aggiornato

  ④ VEDI STATO CORRENTE — in qualsiasi momento:
      python main.py --status

      → mostra holdings, P&L, storico revisioni

─────────────────────────────────────────────────────────────────
FLAGS AVANZATI:
    --state-file PATH        file di stato personalizzato
    --capital N              capitale totale
    --tickers A B C          lista ticker
    --refresh                forza riscaricare dati
    --no-freq-analysis       salta analisi scenari frequenza
    --no-walk-forward        salta walk-forward
    --no-plots               non generare grafici PNG
    --period 10y             periodo storico dati
    --max-stocks 15          numero massimo titoli
    --min-stocks 8           numero minimo titoli
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.downloader import DataDownloader
from analysis.fundamentals import FundamentalAnalyzer
from analysis.technical import TechnicalAnalyzer
from analysis.news import NewsAnalyzer
from analysis.scorer import StockScorer
from optimization.portfolio import PortfolioOptimizer
from optimization.portfolio_state import PortfolioStateManager
from output.report import ReportGenerator
from utils.logger import setup_logger
from utils.config import Config


# ─── ARGOMENTI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Portfolio Optimizer v7",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rebalance", action="store_true",
        help="Rebalance: carica stato, ricalcola, mostra ordini")
    mode.add_argument("--update-prices", action="store_true",
        help="Aggiorna solo prezzi correnti (veloce, ~10 sec)")
    mode.add_argument("--status", action="store_true",
        help="Mostra stato corrente del portafoglio salvato")

    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--state-file", type=str, default="portfolio_state.json")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--period", type=str, default="10y")
    parser.add_argument("--max-stocks", type=int, default=15)
    parser.add_argument("--min-stocks", type=int, default=8)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-freq-analysis", action="store_true")
    parser.add_argument("--no-walk-forward", action="store_true")
    parser.add_argument("--multi-horizon", action="store_true")
    parser.add_argument("--wf-train-years", type=int, default=None)
    parser.add_argument("--wf-test-years",  type=int, default=None)

    return parser.parse_args()


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _header(title, capital, mode):
    print("\n" + "═"*62)
    print(f"  📊  PORTFOLIO OPTIMIZER v7  —  {title}")
    print(f"  Avviato: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    if capital:
        print(f"  Capitale: {capital:,.0f} €")
    print(f"  Modalità: {mode}")
    print("═"*62 + "\n")


def _full_pipeline(args, config, logger, capital, current_holdings=None):
    """Download → analisi → scoring → ottimizzazione. Usato da prima esecuzione e rebalance."""

    print("📥 [1/6] Download dati di mercato...")
    tickers = args.tickers if args.tickers else config.DEFAULT_TICKERS
    # In rebalance: i ticker attuali entrano sempre nel pool
    if current_holdings:
        tickers = list(dict.fromkeys(list(current_holdings.keys()) + tickers))

    downloader = DataDownloader(period=args.period, refresh=args.refresh)
    price_data, ticker_info = downloader.download_all(tickers)
    if price_data.empty:
        print("❌ Impossibile scaricare i dati."); sys.exit(1)
    valid = [t for t in tickers if t in price_data.columns]
    print(f"   ✓ {len(valid)}/{len(tickers)} ticker\n")

    print("📊 [2/6] Analisi tecnica...")
    tech = TechnicalAnalyzer(price_data).analyze_all(valid)
    print(f"   ✓ {len(tech)} titoli\n")

    print("🔍 [3/6] Analisi fondamentale...")
    fund = FundamentalAnalyzer(ticker_info).analyze_all(valid)
    print(f"   ✓ {len(fund)} titoli\n")

    print("⭐ [4/6] Pre-selezione candidati...")
    pre_scorer = StockScorer(tech, fund, ticker_info)
    pre_ranked = pre_scorer.rank_stocks()
    pool_size  = min(args.max_stocks * 2, len(pre_ranked))
    pool       = pre_ranked.head(pool_size)["ticker"].tolist()
    if current_holdings:
        for t in current_holdings:
            if t not in pool and t in valid:
                pool.append(t)
    print(f"   ✓ {len(pool)} candidati\n")

    print("📰 [5/6] Analisi notizie e sentiment...")
    news = NewsAnalyzer(ticker_info).analyze_tickers(pool)
    pos  = sum(1 for n in news.values() if n.get("signal") == "POSITIVO")
    neg  = sum(1 for n in news.values() if n.get("signal") == "NEGATIVO")
    print(f"   ✓ 🟢 {pos}  ⚪ {len(news)-pos-neg}  🔴 {neg}\n")

    print("⚖️  [6/6] Scoring + ottimizzazione...")
    scorer    = StockScorer(tech, fund, ticker_info, news)
    ranked    = scorer.rank_stocks()
    selected  = scorer.select_top_stocks(ranked, min_stocks=args.min_stocks, max_stocks=args.max_stocks)
    print(f"   ✓ {len(selected)} titoli selezionati\n")

    sel_prices = price_data[selected]
    optimizer  = PortfolioOptimizer(price_data=sel_prices, scores=ranked, capital=capital)
    portfolio  = optimizer.optimize()
    backtest   = optimizer.backtest(portfolio)
    print(f"   ✓ Rendimento atteso: {portfolio['expected_return']:.1%}/anno  "
          f"Sharpe: {portfolio['sharpe_ratio']:.2f}\n")

    return portfolio, backtest, price_data, ticker_info, ranked, sel_prices


def _v7_modules(args, config, logger, portfolio, sel_prices):
    """Moduli avanzati v7: freq analysis e walk-forward."""
    optimal_freq = None
    wf_result    = None

    if config.ENABLE_FREQ_ANALYSIS and not args.no_freq_analysis:
        print("📅 [7] Analisi scenari frequenza...")
        try:
            from optimization.frequency_analyzer import FrequencyScenarioAnalyzer
            fa  = FrequencyScenarioAnalyzer(sel_prices, transaction_cost=config.TRANSACTION_COST_PCT,
                                             capital=10000, min_train_days=config.FREQ_MIN_TRAIN_DAYS)
            fdf = fa.run_all_scenarios()
            if not fdf.empty:
                optimal_freq = fa.suggest_optimal(fdf)
                fa.print_report(fdf, optimal_freq)
        except Exception as e:
            print(f"   ⚠️  {e}\n")
    else:
        print("📅 [7] Freq analysis: SALTATA\n")

    if config.ENABLE_WALK_FORWARD and not args.no_walk_forward:
        print("🔬 [8] Walk-forward validation...")
        try:
            from optimization.walk_forward import WalkForwardValidator
            wf = WalkForwardValidator(sel_prices,
                                     train_years=args.wf_train_years or config.WF_TRAIN_YEARS,
                                     test_years=args.wf_test_years or config.WF_TEST_YEARS,
                                     step_months=config.WF_STEP_MONTHS)
            if args.multi_horizon:
                mh = wf.run_multi_horizon()
                bhy = mh.get("best_train_years", 3)
                wf_result = mh["per_horizon_results"].get(bhy)
                print(f"   ✓ Best horizon: {bhy}y")
            else:
                wf_result = wf.run(expanding=config.WF_EXPANDING)
            if wf_result and wf_result.get("n_folds", 0) > 0:
                cs = wf_result["consistency_score"]
                ns = wf_result["aggregate"].get("mean_sharpe", 0)
                pf = wf_result["robustness"].get("pct_positive_folds", 0)
                print(f"   ✓ {wf_result['n_folds']} fold | Consistency={cs:.2f} | "
                      f"Sharpe OOS={ns:.2f} | Fold+={pf:.0%}\n")
                wf.print_report(wf_result)
        except Exception as e:
            print(f"   ⚠️  {e}\n")
    else:
        print("🔬 [8] Walk-forward: SALTATO\n")

    return optimal_freq, wf_result


def _smart_rebalance(config, portfolio, price_data, capital, current_holdings):
    """Calcola ordini di ribilanciamento."""
    try:
        from optimization.smart_rebalancer import SmartRebalancer
        prices = {t: float(price_data[t].dropna().iloc[-1])
                  for t in price_data.columns if not price_data[t].isna().all()}
        rb = SmartRebalancer(tolerance=config.REBALANCE_TOLERANCE,
                             min_trade_eur=config.MIN_TRADE_EUR,
                             tax_rate=config.TAX_RATE_GAINS,
                             buy_cost_eur=config.BUY_COST_EUR,
                             sell_cost_eur=config.SELL_COST_EUR,
                             loss_harvest=config.LOSS_HARVEST)
        orders = rb.compute_trades(current_holdings, portfolio["weights"], capital, prices)
        plan   = rb.summarize(orders, current_holdings)
        rb.print_summary(orders, current_holdings)
        return orders, plan
    except Exception as e:
        print(f"   ⚠️  SmartRebalancer: {e}\n")
        return [], None


def _report(args, portfolio, backtest, price_data, ticker_info, ranked, capital):
    r = ReportGenerator(portfolio=portfolio, backtest=backtest, price_data=price_data,
                        ticker_info=ticker_info, scores=ranked, capital=capital)
    r.print_summary()
    if not args.no_plots:
        try: r.generate_plots(); print("📊 Grafici → output/")
        except Exception: pass
    r.save_report()
    print("✅ Report HTML → output/portfolio_report.html\n")


def _final_summary(portfolio, optimal_freq, wf_result, orders, state_file):
    print("═"*62)
    print("  📋 RIEPILOGO FINALE")
    print("═"*62)
    print(f"  📅 Prossima revisione:  {portfolio['next_review']}")
    print(f"  ⏱  Orizzonte:          {portfolio.get('horizon_label','')}")
    if optimal_freq:
        print(f"  🏆 Freq. ottimale:     {optimal_freq['frequency']} "
              f"(Net Sharpe: {optimal_freq['net_sharpe']:.2f})")
    if wf_result and wf_result.get("n_folds", 0) > 0:
        print(f"  🔬 Walk-forward:       Consistency={wf_result['consistency_score']:.2f} | "
              f"Sharpe OOS={wf_result['aggregate'].get('mean_sharpe',0):.2f}")
    if orders:
        sells = sum(1 for o in orders if o.action == "SELL")
        buys  = sum(1 for o in orders if o.action == "BUY")
        print(f"  🔄 Ordini:             {sells} SELL + {buys} BUY")

    print(f"\n  💾 Stato salvato in:   {state_file}")
    print(f"\n  📌 PROSSIMI PASSI — al prossimo rebalance ({portfolio['next_review']}):")
    print(f"     1.  python main.py --update-prices")
    print(f"         → aggiorna prezzi, mostra P&L attuale")
    print(f"     2.  python main.py --rebalance --capital <capitale_attuale>")
    print(f"         → ricalcola, mostra ordini da eseguire")
    print(f"\n     Oppure in qualsiasi momento:")
    print(f"     python main.py --status   (stato portafoglio)")
    print("═"*62 + "\n")


# ─── COMANDI ─────────────────────────────────────────────────────────────────

def cmd_status(args, config, logger):
    mgr = PortfolioStateManager(args.state_file)
    if not mgr.exists():
        print(f"\n  ⚠️  {args.state_file} non trovato.")
        print("  Crea prima il portafoglio:  python main.py --capital 10000\n")
        sys.exit(0)
    mgr.print_state()


def cmd_update_prices(args, config, logger):
    mgr = PortfolioStateManager(args.state_file)
    if not mgr.exists():
        print(f"\n  ⚠️  {args.state_file} non trovato.")
        print("  Crea prima il portafoglio:  python main.py --capital 10000\n")
        sys.exit(1)
    _header("Aggiornamento Prezzi", None, "--update-prices")
    mgr.update_prices(verbose=True)
    print("  Ora puoi eseguire il rebalance:")
    print("     python main.py --rebalance --capital <capitale>\n")


def cmd_rebalance(args, config, logger):
    mgr = PortfolioStateManager(args.state_file)
    if not mgr.exists():
        print(f"\n  ⚠️  {args.state_file} non trovato.")
        print("  Crea prima il portafoglio:  python main.py --capital 10000\n")
        sys.exit(1)

    current_holdings, meta = mgr.load()
    if not current_holdings:
        print("  ⚠️  Nessuna holding nel file di stato."); sys.exit(1)

    capital = args.capital or float(meta.get("capital", 10000))
    _header("Rebalance Portafoglio", capital, "--rebalance")
    print(f"  📂 Stato: {args.state_file}")
    print(f"  📅 Ultimo rebalance: {meta.get('last_rebalance', 'N/D')}")
    print(f"  📈 Revisione N°:     {int(meta.get('n_rebalances',0))+1}")
    print(f"  📦 Holdings attuali: {len(current_holdings)} titoli\n")
    mgr.print_state()

    portfolio, backtest, price_data, ticker_info, ranked, sel_prices = _full_pipeline(
        args, config, logger, capital, current_holdings=current_holdings)

    optimal_freq, wf_result = _v7_modules(args, config, logger, portfolio, sel_prices)

    print("🔄 [9] Calcolo ordini di ribilanciamento...")
    orders, plan = _smart_rebalance(config, portfolio, price_data, capital, current_holdings)

    _report(args, portfolio, backtest, price_data, ticker_info, ranked, capital)

    state_path = mgr.export_after_optimization(
        portfolio=portfolio, price_data=price_data,
        capital=capital, rebalance_type="REBALANCE", orders=orders)
    print(f"💾 Stato aggiornato → {state_path}\n")

    _final_summary(portfolio, optimal_freq, wf_result, orders, state_path)


def cmd_first_run(args, config, logger):
    capital = args.capital or 10000.0
    _header("Prima Analisi — Creazione Portafoglio", capital, "INIZIALE")

    mgr = PortfolioStateManager(args.state_file)
    if mgr.exists():
        meta = mgr.get_meta()
        print(f"  ⚠️  Esiste già un file di stato: {args.state_file}")
        print(f"     Ultimo rebalance:   {meta.get('last_rebalance', 'N/D')}")
        print(f"     Prossima revisione: {meta.get('next_review', 'N/D')}\n")
        resp = input("  Sovrascrivere e creare nuovo portafoglio? [s/N] ").strip().lower()
        if resp not in ("s", "si", "sì", "y", "yes"):
            print("\n  → Usa  python main.py --rebalance  per aggiornare il portafoglio.\n")
            sys.exit(0)
        print()

    portfolio, backtest, price_data, ticker_info, ranked, sel_prices = _full_pipeline(
        args, config, logger, capital)

    optimal_freq, wf_result = _v7_modules(args, config, logger, portfolio, sel_prices)

    _report(args, portfolio, backtest, price_data, ticker_info, ranked, capital)

    state_path = mgr.export_after_optimization(
        portfolio=portfolio, price_data=price_data,
        capital=capital, rebalance_type="INITIAL")
    print(f"💾 Stato salvato → {state_path}\n")

    _final_summary(portfolio, optimal_freq, wf_result, [], state_path)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    logger = setup_logger()
    config = Config()

    if args.status:
        cmd_status(args, config, logger)
    elif args.update_prices:
        cmd_update_prices(args, config, logger)
    elif args.rebalance:
        cmd_rebalance(args, config, logger)
    else:
        cmd_first_run(args, config, logger)


if __name__ == "__main__":
    main()
