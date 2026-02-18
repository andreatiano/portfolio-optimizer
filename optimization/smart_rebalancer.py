"""
╔══════════════════════════════════════════════════════════════╗
║              SMART REBALANCER — v7                           ║
║  Revisione intelligente del portafoglio con stato precedente ║
╚══════════════════════════════════════════════════════════════╝

Calcola solo le operazioni REALMENTE necessarie per passare
dal portafoglio attuale al target ottimizzato, minimizzando:
  - Numero di transazioni
  - Costi di trading
  - Impatto fiscale stimato

Utilizzo:
    rebalancer = SmartRebalancer()
    current = {
        'AAPL': Holding('AAPL', quantity=10, avg_price=150, current_price=180),
        'MSFT': Holding('MSFT', quantity=5,  avg_price=300, current_price=420),
    }
    target_weights = {'AAPL': 0.20, 'MSFT': 0.15, 'NVDA': 0.10, ...}
    orders = rebalancer.compute_trades(current, target_weights, total_capital=10000, current_prices={...})
    summary = rebalancer.summarize(orders)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("portfolio_optimizer")


# ─── DATACLASSES ─────────────────────────────────────────────────────────────

@dataclass
class Holding:
    """
    Posizione attualmente detenuta nel portafoglio.

    Args:
        ticker:        Simbolo del titolo
        quantity:      Numero di azioni/quote detenute
        avg_price:     Prezzo medio di carico (EUR/USD)
        current_price: Prezzo attuale di mercato
    """
    ticker:        str
    quantity:      float
    avg_price:     float
    current_price: float

    @property
    def market_value(self) -> float:
        """Valore corrente di mercato della posizione."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Costo totale di acquisto (base imponibile)."""
        return self.quantity * self.avg_price

    @property
    def unrealized_gain(self) -> float:
        """Plusvalenza/minusvalenza non realizzata in EUR."""
        return (self.current_price - self.avg_price) * self.quantity

    @property
    def unrealized_gain_pct(self) -> float:
        """Plusvalenza non realizzata in % sul costo."""
        if self.cost_basis <= 0:
            return 0.0
        return self.unrealized_gain / self.cost_basis

    @property
    def is_in_gain(self) -> bool:
        return self.current_price > self.avg_price

    @property
    def is_in_loss(self) -> bool:
        return self.current_price < self.avg_price


@dataclass
class TradeOrder:
    """
    Ordine di acquisto o vendita da eseguire.

    Campi:
        action:    'BUY' | 'SELL' | 'HOLD'
        priority:  1=urgente (delta>5%), 2=normale (delta 1-5%), 3=opzionale (<1%)
        reason:    Motivazione leggibile dell'operazione
    """
    ticker:           str
    action:           str          # 'BUY' | 'SELL' | 'HOLD'
    quantity:         float        # Numero azioni da comprare/vendere
    estimated_price:  float        # Prezzo stimato di esecuzione
    amount_eur:       float        # Controvalore in EUR
    transaction_cost: float        # Commissione stimata (EUR)
    tax_impact:       float        # Imposta plusvalenze stimata (solo SELL in gain)
    net_cost:         float        # Costo totale = amount + commissione + tasse
    delta_weight:     float        # Delta peso (target - current)
    current_weight:   float        # Peso attuale nel portafoglio
    target_weight:    float        # Peso target
    priority:         int          # 1, 2, 3
    reason:           str          # Motivazione
    is_new_position:  bool = False # True se ticker non era in portafoglio
    is_full_exit:     bool = False # True se vende tutta la posizione

    @property
    def total_impact(self) -> float:
        """Impatto totale incluse tasse e commissioni."""
        return self.transaction_cost + self.tax_impact


# ─── SMART REBALANCER ────────────────────────────────────────────────────────

class SmartRebalancer:
    """
    Calcola le operazioni minimali di ribilanciamento tra portafoglio
    attuale e portafoglio target.

    Parametri configurabili (tutti override-abili nel costruttore):
        TOLERANCE:       Delta peso minimo prima di agire (default ±0.8%)
        MIN_TRADE_EUR:   Importo minimo per singola operazione
        TAX_RATE:        Aliquota plusvalenze (IT: 26%)
        BUY_COST_EUR:    Commissione fissa acquisto
        SELL_COST_EUR:   Commissione fissa vendita
        LOSS_HARVEST:    Se True, privilegia vendite in perdita per compensazione fiscale
    """

    # Parametri default
    TOLERANCE       = 0.008   # ±0.8%: sotto questa soglia non si agisce
    MIN_TRADE_EUR   = 50.0    # Importo minimo per operazione
    TAX_RATE        = 0.26    # Aliquota plusvalenze IT
    BUY_COST_EUR    = 1.0     # Commissione acquisto (EUR fisso)
    SELL_COST_EUR   = 1.0     # Commissione vendita (EUR fisso)
    LOSS_HARVEST    = True    # Privilegia minusvalenze per compensazione

    def __init__(self, **kwargs):
        """Permette override dei parametri default."""
        for k, v in kwargs.items():
            if hasattr(self, k.upper()):
                setattr(self, k.upper(), v)
            else:
                raise ValueError(f"Parametro sconosciuto: {k}")

    # ─── METODO PRINCIPALE ───────────────────────────────────────────────────

    def compute_trades(
        self,
        current_holdings:  Dict[str, Holding],
        target_weights:    Dict[str, float],
        total_capital:     float,
        current_prices:    Dict[str, float],
    ) -> List[TradeOrder]:
        """
        Calcola la lista ottimizzata di ordini per ribilanciare il portafoglio.

        Args:
            current_holdings: Dict {ticker: Holding} con posizioni correnti
            target_weights:   Dict {ticker: float} pesi target (somma ≈ 1)
            total_capital:    Capitale totale del portafoglio (EUR)
            current_prices:   Dict {ticker: float} prezzi correnti

        Returns:
            Lista di TradeOrder ordinata: SELL (minusvalenze) → SELL (plusvalenze) → BUY
        """
        if not target_weights:
            logger.warning("target_weights vuoto, nessuna operazione calcolata")
            return []

        # Normalizza pesi target (devono sommare a 1)
        total_w = sum(target_weights.values())
        if abs(total_w - 1.0) > 0.01:
            logger.warning(f"Pesi target sommano a {total_w:.3f}, normalizzo")
            target_weights = {t: w / total_w for t, w in target_weights.items()}

        # Calcola valore corrente per ogni ticker
        current_values: Dict[str, float] = {}
        for ticker, holding in current_holdings.items():
            price = current_prices.get(ticker, holding.current_price)
            current_values[ticker] = holding.quantity * price

        # Valore totale corrente del portafoglio
        portfolio_value = sum(current_values.values())
        if portfolio_value <= 0:
            portfolio_value = total_capital

        # Tutti i ticker coinvolti (union di correnti e target)
        all_tickers = set(current_holdings.keys()) | set(target_weights.keys())

        orders: List[TradeOrder] = []

        for ticker in all_tickers:
            price = current_prices.get(ticker)
            if price is None or price <= 0:
                logger.debug(f"Prezzo non disponibile per {ticker}, salto")
                continue

            # Pesi correnti e target
            current_val = current_values.get(ticker, 0.0)
            current_w   = current_val / portfolio_value if portfolio_value > 0 else 0.0
            target_w    = target_weights.get(ticker, 0.0)

            delta_w   = target_w - current_w
            delta_val = delta_w * total_capital

            # Filtra delta troppo piccoli
            if abs(delta_w) < self.TOLERANCE:
                logger.debug(f"{ticker}: delta {delta_w:.3%} < tolleranza, HOLD")
                continue

            # Filtra importi troppo piccoli
            if abs(delta_val) < self.MIN_TRADE_EUR:
                logger.debug(f"{ticker}: delta_val {delta_val:.0f}€ < minimo, HOLD")
                continue

            action   = "BUY" if delta_val > 0 else "SELL"
            quantity = abs(delta_val) / price
            holding  = current_holdings.get(ticker)

            # Calcola impatto fiscale (solo per vendite)
            tax      = self._estimate_tax(holding, quantity, action, price)
            cost_eur = self.BUY_COST_EUR if action == "BUY" else self.SELL_COST_EUR

            # Determina se è uscita completa
            is_full_exit = (
                action == "SELL"
                and holding is not None
                and quantity >= holding.quantity * 0.95
            )
            if is_full_exit:
                quantity = holding.quantity  # vendi tutto
                delta_val = quantity * price

            is_new = ticker not in current_holdings and action == "BUY"

            order = TradeOrder(
                ticker           = ticker,
                action           = action,
                quantity         = quantity,
                estimated_price  = price,
                amount_eur       = abs(delta_val),
                transaction_cost = cost_eur,
                tax_impact       = tax,
                net_cost         = abs(delta_val) + cost_eur + tax,
                delta_weight     = delta_w,
                current_weight   = current_w,
                target_weight    = target_w,
                priority         = self._priority(delta_w),
                reason           = self._reason(ticker, delta_w, holding, is_new),
                is_new_position  = is_new,
                is_full_exit     = is_full_exit,
            )
            orders.append(order)

        # Ordina in modo ottimale: prima le vendite, poi gli acquisti
        orders = self._sort_orders(orders, current_holdings)
        logger.info(f"SmartRebalancer: {len(orders)} ordini generati "
                    f"({sum(1 for o in orders if o.action=='SELL')} SELL, "
                    f"{sum(1 for o in orders if o.action=='BUY')} BUY)")
        return orders

    # ─── SOMMARIO ────────────────────────────────────────────────────────────

    def summarize(
        self,
        orders: List[TradeOrder],
        current_holdings: Optional[Dict[str, Holding]] = None
    ) -> dict:
        """
        Produce un sommario leggibile degli ordini.

        Returns:
            Dict con: n_orders, n_buy, n_sell, total_volume, total_costs,
                      total_tax, net_impact, orders_df, recommendations
        """
        if not orders:
            return {
                "n_orders": 0,
                "n_buy": 0,
                "n_sell": 0,
                "total_volume_eur": 0,
                "total_costs_eur": 0,
                "total_tax_eur": 0,
                "total_impact_eur": 0,
                "orders_df": pd.DataFrame(),
                "recommendations": ["✅ Portafoglio già allineato al target, nessuna operazione necessaria."],
            }

        sells = [o for o in orders if o.action == "SELL"]
        buys  = [o for o in orders if o.action == "BUY"]

        total_volume = sum(o.amount_eur for o in orders)
        total_costs  = sum(o.transaction_cost for o in orders)
        total_tax    = sum(o.tax_impact for o in orders)
        total_impact = total_costs + total_tax

        # Liquidity from sells
        sell_proceeds = sum(o.amount_eur - o.transaction_cost - o.tax_impact for o in sells)
        buy_needed    = sum(o.amount_eur + o.transaction_cost for o in buys)

        # DataFrame ordini
        rows = []
        for o in orders:
            rows.append({
                "Ticker":        o.ticker,
                "Azione":        o.action,
                "Quantità":      round(o.quantity, 4),
                "Prezzo (€)":    round(o.estimated_price, 2),
                "Importo (€)":   round(o.amount_eur, 2),
                "Commissione":   round(o.transaction_cost, 2),
                "Tasse est. (€)": round(o.tax_impact, 2),
                "Δ Peso":        f"{o.delta_weight:+.1%}",
                "Peso Att.":     f"{o.current_weight:.1%}",
                "Peso Target":   f"{o.target_weight:.1%}",
                "Priorità":      "🔴" if o.priority == 1 else "🟡" if o.priority == 2 else "🟢",
                "Note":          o.reason,
            })
        df = pd.DataFrame(rows)

        # Raccomandazioni intelligenti
        recs = self._build_recommendations(orders, current_holdings or {})

        return {
            "n_orders":        len(orders),
            "n_buy":           len(buys),
            "n_sell":          len(sells),
            "total_volume_eur": round(total_volume, 2),
            "total_costs_eur":  round(total_costs, 2),
            "total_tax_eur":    round(total_tax, 2),
            "total_impact_eur": round(total_impact, 2),
            "sell_proceeds_eur": round(sell_proceeds, 2),
            "buy_needed_eur":   round(buy_needed, 2),
            "cash_balance":     round(sell_proceeds - buy_needed, 2),
            "orders_df":        df,
            "recommendations":  recs,
        }

    def print_summary(self, orders: List[TradeOrder],
                      current_holdings: Optional[Dict[str, Holding]] = None):
        """Stampa il sommario formattato su console."""
        s = self.summarize(orders, current_holdings)

        print("\n" + "═"*65)
        print("  🔄  PIANO DI RIBILANCIAMENTO INTELLIGENTE")
        print("═"*65)
        print(f"  Operazioni totali:   {s['n_orders']}  "
              f"({s['n_sell']} SELL  |  {s['n_buy']} BUY)")
        print(f"  Volume totale:       {s['total_volume_eur']:>10,.0f} €")
        print(f"  Commissioni stimate: {s['total_costs_eur']:>10,.2f} €")
        print(f"  Tasse stimate:       {s['total_tax_eur']:>10,.2f} €")
        print(f"  Impatto totale:      {s['total_impact_eur']:>10,.2f} €")
        print(f"  Liquidità SELL:      {s['sell_proceeds_eur']:>10,.0f} €")
        print(f"  Fabbisogno BUY:      {s['buy_needed_eur']:>10,.0f} €")
        balance = s['cash_balance']
        sign = "+" if balance >= 0 else ""
        print(f"  Saldo cassa:         {sign}{balance:>9,.0f} €")

        if not s['orders_df'].empty:
            print(f"\n  {'Ticker':<8} {'Azione':<6} {'Qty':>7} {'€':>8} {'Tasse':>7} {'ΔPeso':>7}  Note")
            print("  " + "─"*63)
            for o in orders:
                print(f"  {o.ticker:<8} {o.action:<6} {o.quantity:>7.2f} "
                      f"{o.amount_eur:>8,.0f} {o.tax_impact:>7,.0f}"
                      f" {o.delta_weight:>+7.1%}  {o.reason[:28]}")

        print("\n  💡 RACCOMANDAZIONI")
        print("  " + "─"*63)
        for rec in s["recommendations"]:
            print(f"  {rec}")
        print("═"*65 + "\n")

    # ─── METODI PRIVATI ──────────────────────────────────────────────────────

    def _estimate_tax(
        self,
        holding: Optional[Holding],
        quantity: float,
        action: str,
        current_price: float,
    ) -> float:
        """
        Stima l'imposta sulle plusvalenze realizzate.

        Formula: (prezzo_corrente - prezzo_medio) × qty_venduta × TAX_RATE
        Restituisce 0 se la posizione è in perdita (minusvalenza).
        """
        if action != "SELL" or holding is None:
            return 0.0

        gain_per_share = current_price - holding.avg_price
        if gain_per_share <= 0:
            return 0.0  # Minusvalenza: nessuna imposta

        # Non si può vendere più di quello che si ha
        qty_sellable = min(quantity, holding.quantity)
        taxable_gain = gain_per_share * qty_sellable
        return round(taxable_gain * self.TAX_RATE, 2)

    def _priority(self, delta_w: float) -> int:
        """
        Priorità dell'ordine in base all'entità del delta:
          1 = urgente (|delta| > 5%)
          2 = normale  (|delta| 1–5%)
          3 = opzionale (|delta| < 1%)
        """
        adw = abs(delta_w)
        if adw > 0.05:
            return 1
        elif adw > 0.01:
            return 2
        return 3

    def _reason(
        self,
        ticker: str,
        delta_w: float,
        holding: Optional[Holding],
        is_new: bool,
    ) -> str:
        """Costruisce una descrizione leggibile dell'operazione."""
        adw = abs(delta_w)
        if is_new:
            return f"Nuova posizione ({adw:.1%} del portafoglio)"
        if holding is None and delta_w < 0:
            return "Ticker non presente, nessuna azione SELL"

        if delta_w > 0:
            return f"Sottopesato: incremento +{adw:.1%}"
        else:
            if holding and holding.is_in_loss:
                pct = holding.unrealized_gain_pct
                return f"Sovrappesato (in perdita {pct:.1%}): loss harvesting"
            elif holding and holding.is_in_gain:
                pct = holding.unrealized_gain_pct
                return f"Sovrappesato (in gain {pct:.1%}): riduzione parziale"
            return f"Sovrappesato: riduzione -{adw:.1%}"

    def _sort_orders(
        self,
        orders: List[TradeOrder],
        current_holdings: Dict[str, Holding],
    ) -> List[TradeOrder]:
        """
        Ordina gli ordini in modo ottimale:
          1. SELL in perdita (loss harvesting — compensano plusvalenze)
          2. SELL in guadagno (da più grande a più piccolo per liquidità)
          3. BUY (da delta più grande a più piccolo)
        """
        sells_loss = sorted(
            [o for o in orders if o.action == "SELL" and o.tax_impact == 0],
            key=lambda o: o.amount_eur, reverse=True
        )
        sells_gain = sorted(
            [o for o in orders if o.action == "SELL" and o.tax_impact > 0],
            key=lambda o: o.amount_eur, reverse=True
        )
        buys = sorted(
            [o for o in orders if o.action == "BUY"],
            key=lambda o: abs(o.delta_weight), reverse=True
        )

        return sells_loss + sells_gain + buys

    def _build_recommendations(
        self,
        orders: List[TradeOrder],
        current_holdings: Dict[str, Holding],
    ) -> List[str]:
        """Genera raccomandazioni intelligenti basate sul piano di ordini."""
        recs = []

        sells = [o for o in orders if o.action == "SELL"]
        buys  = [o for o in orders if o.action == "BUY"]

        # Copertura liquidità
        sell_proceeds = sum(o.amount_eur - o.transaction_cost - o.tax_impact for o in sells)
        buy_needed    = sum(o.amount_eur + o.transaction_cost for o in buys)
        balance = sell_proceeds - buy_needed
        if balance < 0:
            recs.append(f"⚠️  Servono {abs(balance):,.0f}€ di liquidità aggiuntiva per i BUY")
        elif balance > 100:
            recs.append(f"💵 Rimangono {balance:,.0f}€ di cassa dopo il ribilanciamento")

        # Tasse
        total_tax = sum(o.tax_impact for o in orders)
        loss_val  = sum(o.amount_eur for o in sells if o.tax_impact == 0 and
                        current_holdings.get(o.ticker) and current_holdings[o.ticker].is_in_loss)
        if total_tax > 50:
            recs.append(f"💸 Imposta plusvalenze stimata: {total_tax:,.0f}€ — considera la compensazione con minusvalenze")
        if loss_val > 0:
            recs.append(f"🎯 Loss harvesting attivo: {loss_val:,.0f}€ di minusvalenze da realizzare")

        # Titoli nuovi
        new_pos = [o for o in buys if o.is_new_position]
        if new_pos:
            recs.append(f"🆕 Nuove posizioni da aprire: {', '.join(o.ticker for o in new_pos)}")

        # Uscite complete
        exits = [o for o in sells if o.is_full_exit]
        if exits:
            recs.append(f"🚪 Uscita completa da: {', '.join(o.ticker for o in exits)}")

        # Ordine esecuzione
        recs.append("📋 Sequenza consigliata: eseguire prima le SELL per liberare liquidità, poi i BUY")

        if not recs:
            recs.append("✅ Piano di ribilanciamento ottimale, nessuna criticità rilevata")

        return recs
