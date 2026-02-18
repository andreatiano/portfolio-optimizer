"""
╔══════════════════════════════════════════════════════════════╗
║           PORTFOLIO STATE MANAGER — v7                       ║
║     Salvataggio e caricamento dello stato del portafoglio    ║
╚══════════════════════════════════════════════════════════════╝

Gestisce il ciclo di vita completo del portafoglio tra una
revisione e la successiva:

  1. EXPORT: dopo ogni ottimizzazione, salva lo stato in JSON
             con quantità, prezzi medi, data acquisto, storico
  2. LOAD:   al momento del rebalance, ricarica lo stato
  3. UPDATE: aggiorna i prezzi correnti da yfinance senza
             rieseguire l'ottimizzazione completa
  4. HISTORY: tiene traccia di tutte le revisioni passate

Flusso tipico:
  ─────────────────────────────────────────────────────────
  Mese 0:  python main.py --capital 10000
           → genera portafoglio ottimizzato
           → salva automaticamente portfolio_state.json

  Mese 6:  python main.py --update-prices portfolio_state.json
           → aggiorna i prezzi correnti nel file

           python main.py --rebalance portfolio_state.json --capital 10000
           → carica stato, ricalcola ottimizzazione, mostra ordini
           → aggiorna portfolio_state.json con nuovo stato
  ─────────────────────────────────────────────────────────

Formato portfolio_state.json:
  {
    "meta": {
      "version": "7",
      "created": "18/02/2026 17:30",
      "last_updated": "18/02/2026 17:30",
      "last_rebalance": "18/02/2026 17:30",
      "next_review": "01/08/2026",
      "capital": 10000,
      "n_rebalances": 1
    },
    "holdings": {
      "AAPL": {
        "quantity": 5.26,
        "avg_price": 190.00,
        "current_price": 195.00,
        "weight_target": 0.10,
        "amount_invested": 1000.0,
        "first_buy_date": "18/02/2026",
        "last_trade_date": "18/02/2026",
        "notes": ""
      },
      ...
    },
    "history": [
      {
        "date": "18/02/2026 17:30",
        "type": "INITIAL",
        "capital": 10000,
        "n_holdings": 12,
        "expected_return": 0.18,
        "sharpe_ratio": 1.4,
        "tickers": ["AAPL", "MSFT", ...]
      }
    ]
  }
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("portfolio_optimizer")

DEFAULT_STATE_FILE = "portfolio_state.json"


# ─── GESTIONE STATO ──────────────────────────────────────────────────────────

class PortfolioStateManager:
    """
    Gestisce la persistenza dello stato del portafoglio tra revisioni.

    Permette di:
    - Esportare il portafoglio ottimizzato in JSON dopo ogni run
    - Ricaricare lo stato al momento del rebalance successivo
    - Aggiornare i prezzi correnti senza rieseguire l'ottimizzazione
    - Tracciare lo storico di tutte le revisioni
    """

    VERSION = "7"

    def __init__(self, filepath: str = DEFAULT_STATE_FILE):
        self.filepath = filepath

    # ─── EXPORT ──────────────────────────────────────────────────────────────

    def export_after_optimization(
        self,
        portfolio:       dict,
        price_data,           # pd.DataFrame
        capital:         float,
        rebalance_type:  str = "INITIAL",
        orders:          Optional[List] = None,
    ) -> str:
        """
        Salva lo stato completo del portafoglio dopo un'ottimizzazione.

        Args:
            portfolio:      Output di PortfolioOptimizer.optimize()
            price_data:     DataFrame prezzi (per ricavare prezzo corrente)
            capital:        Capitale totale
            rebalance_type: 'INITIAL' | 'REBALANCE' | 'PARTIAL'
            orders:         Lista TradeOrder eseguiti (opzionale)

        Returns:
            Percorso del file salvato
        """
        now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
        today   = datetime.now().strftime("%d/%m/%Y")

        # Carica stato esistente (per preservare history e avg_price)
        existing = self._load_raw()

        # Costruisci holdings
        holdings = {}
        for ticker, weight in portfolio["weights"].items():
            # Prezzo corrente dai dati scaricati
            current_price = None
            if ticker in price_data.columns:
                last_valid = price_data[ticker].dropna()
                if len(last_valid) > 0:
                    current_price = float(last_valid.iloc[-1])

            if current_price is None or current_price <= 0:
                logger.warning(f"Prezzo non disponibile per {ticker}, uso stima")
                current_price = 100.0

            amount     = weight * capital
            quantity   = amount / current_price

            # Avg price: se già in portafoglio, mantieni il prezzo medio storico
            prev_holding = (existing.get("holdings", {}) or {}).get(ticker)
            if prev_holding and rebalance_type != "INITIAL":
                # Calcola nuovo avg_price con media pesata (FIFO approssimato)
                prev_qty  = float(prev_holding.get("quantity", 0))
                prev_avg  = float(prev_holding.get("avg_price", current_price))
                prev_date = prev_holding.get("first_buy_date", today)
                # Se stiamo comprando altro
                if quantity > prev_qty:
                    extra_qty = quantity - prev_qty
                    new_avg   = (prev_avg * prev_qty + current_price * extra_qty) / quantity
                else:
                    new_avg   = prev_avg  # vendita parziale: avg non cambia
                avg_price      = round(new_avg, 4)
                first_buy_date = prev_date
                last_trade     = today
            else:
                avg_price      = round(current_price, 4)
                first_buy_date = today
                last_trade     = today

            holdings[ticker] = {
                "quantity":       round(quantity, 6),
                "avg_price":      avg_price,
                "current_price":  round(current_price, 4),
                "weight_target":  round(weight, 6),
                "amount_invested": round(amount, 2),
                "first_buy_date": first_buy_date,
                "last_trade_date": last_trade,
                "notes":          "",
            }

        # Aggiorna history
        history = list(existing.get("history", []) or [])
        history_entry = {
            "date":            now_str,
            "type":            rebalance_type,
            "capital":         capital,
            "n_holdings":      len(holdings),
            "expected_return": round(portfolio.get("expected_return", 0), 4),
            "expected_vol":    round(portfolio.get("expected_volatility", 0), 4),
            "sharpe_ratio":    round(portfolio.get("sharpe_ratio", 0), 4),
            "horizon_label":   portfolio.get("horizon_label", ""),
            "next_review":     portfolio.get("next_review", ""),
            "tickers":         list(portfolio["weights"].keys()),
        }
        if orders:
            history_entry["n_orders"] = len(orders)
            history_entry["sell_orders"] = sum(1 for o in orders if o.action == "SELL")
            history_entry["buy_orders"]  = sum(1 for o in orders if o.action == "BUY")
        history.append(history_entry)

        # Conta rebalance
        prev_meta   = existing.get("meta", {}) or {}
        n_rebalance = int(prev_meta.get("n_rebalances", 0)) + 1

        state = {
            "meta": {
                "version":        self.VERSION,
                "created":        prev_meta.get("created", now_str),
                "last_updated":   now_str,
                "last_rebalance": now_str,
                "next_review":    portfolio.get("next_review", ""),
                "horizon_label":  portfolio.get("horizon_label", ""),
                "capital":        capital,
                "n_rebalances":   n_rebalance,
                "state_file":     os.path.abspath(self.filepath),
            },
            "holdings": holdings,
            "history":  history,
        }

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        logger.info(f"Stato portafoglio salvato: {self.filepath} ({len(holdings)} holdings)")
        return self.filepath

    # ─── LOAD ────────────────────────────────────────────────────────────────

    def load(self) -> Tuple[Dict, dict]:
        """
        Carica lo stato del portafoglio dal file JSON.

        Returns:
            (holdings_dict, meta_dict)
            holdings_dict: {ticker: Holding}
            meta_dict:     informazioni sul file (data, capitale, ecc.)
        """
        from optimization.smart_rebalancer import Holding

        raw = self._load_raw()
        if not raw:
            return {}, {}

        meta     = raw.get("meta", {})
        raw_hold = raw.get("holdings", {})

        holdings = {}
        for ticker, info in raw_hold.items():
            if ticker.startswith("_"):  # salta commenti
                continue
            try:
                holdings[ticker] = Holding(
                    ticker        = ticker,
                    quantity      = float(info.get("quantity", 0)),
                    avg_price     = float(info.get("avg_price", 0)),
                    current_price = float(info.get("current_price",
                                         info.get("avg_price", 0))),
                )
            except Exception as e:
                logger.warning(f"Holding {ticker} non valido: {e}")

        logger.info(f"Stato caricato: {len(holdings)} holdings "
                    f"(ultimo rebalance: {meta.get('last_rebalance', 'N/D')})")
        return holdings, meta

    # ─── UPDATE PREZZI ───────────────────────────────────────────────────────

    def update_prices(self, verbose: bool = True) -> dict:
        """
        Aggiorna i prezzi correnti nel file di stato scaricandoli da yfinance.
        Non modifica avg_price, quantity o altri campi.

        Returns:
            Dizionario {ticker: new_price}
        """
        raw = self._load_raw()
        if not raw:
            raise FileNotFoundError(f"File di stato non trovato: {self.filepath}")

        holdings = raw.get("holdings", {})
        tickers  = [t for t in holdings if not t.startswith("_")]

        if not tickers:
            print("   ⚠️  Nessun ticker nel file di stato")
            return {}

        print(f"\n📡 Aggiornamento prezzi per {len(tickers)} ticker...")

        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance non installato: pip install yfinance")

        updated = {}
        failed  = []

        # Download batch
        try:
            raw_data = yf.download(
                tickers, period="2d", auto_adjust=True,
                progress=False, threads=True
            )
            if hasattr(raw_data.columns, "levels"):
                closes = raw_data["Close"]
                if hasattr(closes.columns, "tolist"):
                    closes.columns = [str(c) for c in closes.columns]
            else:
                closes = raw_data[["Close"]] if "Close" in raw_data else raw_data

            for ticker in tickers:
                try:
                    if ticker in closes.columns:
                        last = closes[ticker].dropna()
                        if len(last) > 0:
                            new_price = float(last.iloc[-1])
                            holdings[ticker]["current_price"] = round(new_price, 4)
                            updated[ticker] = new_price
                        else:
                            failed.append(ticker)
                    else:
                        failed.append(ticker)
                except Exception:
                    failed.append(ticker)
        except Exception as e:
            logger.warning(f"Batch download fallito, provo singolarmente: {e}")
            for ticker in tickers:
                try:
                    tk   = yf.Ticker(ticker)
                    hist = tk.history(period="2d")
                    if not hist.empty:
                        new_price = float(hist["Close"].iloc[-1])
                        holdings[ticker]["current_price"] = round(new_price, 4)
                        updated[ticker] = new_price
                    else:
                        failed.append(ticker)
                except Exception:
                    failed.append(ticker)

        # Aggiorna timestamp
        raw["meta"]["last_updated"] = datetime.now().strftime("%d/%m/%Y %H:%M")
        raw["holdings"] = holdings

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"\n{'Ticker':<10} {'Prezzo Prev':>13} {'Prezzo Att':>12} {'Var':>8} {'P&L non real.':>15}")
            print("─" * 62)
            for ticker in tickers:
                if ticker not in updated:
                    print(f"  {ticker:<8}  ⚠️  aggiornamento fallito")
                    continue
                info      = holdings[ticker]
                new_price = updated[ticker]
                qty       = float(info.get("quantity", 0))
                avg       = float(info.get("avg_price", new_price))
                var_pct   = (new_price - avg) / avg * 100 if avg > 0 else 0
                pnl       = (new_price - avg) * qty
                color_on  = "\033[92m" if pnl >= 0 else "\033[91m"
                color_off = "\033[0m"
                print(f"  {ticker:<8}  {avg:>12.2f}  {new_price:>11.2f}  "
                      f"{color_on}{var_pct:>+6.1f}%  {pnl:>+13.2f}{color_off}")

            print(f"\n  ✓ Aggiornati: {len(updated)}/{len(tickers)} ticker")
            if failed:
                print(f"  ⚠️  Falliti: {', '.join(failed)}")

            # Riepilogo P&L totale
            total_pnl  = sum((updated.get(t, float(holdings[t].get("current_price", 0))) -
                              float(holdings[t].get("avg_price", 0))) *
                             float(holdings[t].get("quantity", 0))
                             for t in tickers if t in updated)
            total_inv  = sum(float(holdings[t].get("amount_invested", 0)) for t in tickers)
            total_curr = sum(updated.get(t, float(holdings[t].get("current_price",0))) *
                             float(holdings[t].get("quantity",0)) for t in tickers)
            print(f"\n  💰 Valore investito:  {total_inv:>10,.2f} €")
            print(f"  💹 Valore corrente:   {total_curr:>10,.2f} €")
            pnl_color = "\033[92m" if total_pnl >= 0 else "\033[91m"
            print(f"  {'📈' if total_pnl >= 0 else '📉'} P&L non realizzato:  "
                  f"{pnl_color}{total_pnl:>+9,.2f} €  "
                  f"({total_pnl/total_inv*100:>+.1f}%)\033[0m")
            print(f"\n  ✅ File aggiornato: {self.filepath}\n")

        return updated

    # ─── PRINT STATO ─────────────────────────────────────────────────────────

    def print_state(self):
        """Stampa il riepilogo dello stato corrente del portafoglio."""
        raw = self._load_raw()
        if not raw:
            print(f"  ❌ File non trovato: {self.filepath}")
            return

        meta     = raw.get("meta", {})
        holdings = raw.get("holdings", {})
        history  = raw.get("history", [])

        print("\n" + "═"*65)
        print("  📋  STATO CORRENTE PORTAFOGLIO")
        print("═"*65)
        print(f"  File:             {self.filepath}")
        print(f"  Ultimo rebalance: {meta.get('last_rebalance', 'N/D')}")
        print(f"  Prossima revisione: {meta.get('next_review', 'N/D')}")
        print(f"  Capitale:         {meta.get('capital', 0):>10,.0f} €")
        print(f"  N° revisioni:     {meta.get('n_rebalances', 0)}")

        print(f"\n  {'Ticker':<8} {'Qty':>8} {'Avg (€)':>10} {'Att (€)':>10} "
              f"{'P&L':>10} {'%':>7} {'Peso':>7}")
        print("  " + "─"*65)

        total_inv  = 0.0
        total_curr = 0.0
        for ticker, info in holdings.items():
            if ticker.startswith("_"):
                continue
            qty   = float(info.get("quantity", 0))
            avg   = float(info.get("avg_price", 0))
            curr  = float(info.get("current_price", avg))
            w     = float(info.get("weight_target", 0))
            pnl   = (curr - avg) * qty
            pct   = (curr - avg) / avg * 100 if avg > 0 else 0
            inv   = avg * qty
            total_inv  += inv
            total_curr += curr * qty
            color = "\033[92m" if pnl >= 0 else "\033[91m"
            off   = "\033[0m"
            print(f"  {ticker:<8} {qty:>8.2f} {avg:>10.2f} {curr:>10.2f} "
                  f"{color}{pnl:>+9.2f} {pct:>+6.1f}%{off}  {w:>6.1%}")

        total_pnl = total_curr - total_inv
        pnl_pct   = total_pnl / total_inv * 100 if total_inv > 0 else 0
        print("  " + "─"*65)
        color = "\033[92m" if total_pnl >= 0 else "\033[91m"
        print(f"  {'TOTALE':<8} {'':>8} {total_inv:>10,.0f} {total_curr:>10,.0f} "
              f"{color}{total_pnl:>+9.0f} {pnl_pct:>+6.1f}%\033[0m")

        if history:
            print(f"\n  📅  STORICO REVISIONI ({len(history)})")
            print("  " + "─"*65)
            for h in history[-5:]:  # ultime 5
                print(f"  {h.get('date','?'):<18} {h.get('type','?'):<12} "
                      f"Titoli: {h.get('n_holdings',0):<3} "
                      f"CAGR atteso: {h.get('expected_return',0):>5.1%} "
                      f"Sharpe: {h.get('sharpe_ratio',0):.2f}")
        print("═"*65 + "\n")

    # ─── UTILITY INTERNA ─────────────────────────────────────────────────────

    def _load_raw(self) -> dict:
        """Carica il JSON grezzo dal file. Restituisce {} se non esiste."""
        if not os.path.exists(self.filepath):
            return {}
        try:
            with open(self.filepath, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Errore lettura {self.filepath}: {e}")
            return {}

    def exists(self) -> bool:
        return os.path.exists(self.filepath)

    def get_meta(self) -> dict:
        return self._load_raw().get("meta", {})


# ─── FUNZIONI DI CONVENIENZA ─────────────────────────────────────────────────

def export_portfolio_state(
    portfolio:  dict,
    price_data,
    capital:    float,
    filepath:   str = DEFAULT_STATE_FILE,
    rebalance_type: str = "INITIAL",
    orders:     Optional[List] = None,
) -> str:
    """Shortcut per esportare lo stato dopo un'ottimizzazione."""
    mgr = PortfolioStateManager(filepath)
    return mgr.export_after_optimization(
        portfolio, price_data, capital, rebalance_type, orders
    )


def load_portfolio_state(filepath: str = DEFAULT_STATE_FILE) -> Tuple[Dict, dict]:
    """Shortcut per caricare lo stato del portafoglio."""
    mgr = PortfolioStateManager(filepath)
    return mgr.load()


def update_portfolio_prices(filepath: str = DEFAULT_STATE_FILE) -> dict:
    """Shortcut per aggiornare i prezzi correnti."""
    mgr = PortfolioStateManager(filepath)
    return mgr.update_prices()
