"""
Modulo di analisi delle notizie recenti per ogni titolo selezionato.

Funzionamento:
  1. Scarica le ultime notizie da yfinance (gratis, nessuna API key)
  2. Calcola un sentiment score con analisi keyword (no dipendenze extra)
  3. Produce un news_score 0-100 e un segnale: POSITIVO / NEUTRO / NEGATIVO
  4. Identifica i temi principali: earnings, M&A, regolatori, prodotti, macro

Il news_score viene poi usato dal StockScorer come componente aggiuntiva
del punteggio composito finale (peso ~10%).
"""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger("portfolio_optimizer")


# ─── DIZIONARI SENTIMENT ─────────────────────────────────────────────────────

STRONG_POSITIVE = [
    "beat", "beats", "record", "record-high", "surge", "surges", "soar", "soars",
    "breakthrough", "blockbuster", "blowout", "outperform", "outperforms",
    "upgrade", "upgraded", "strong buy", "buy rating", "raise target",
    "raised guidance", "raised outlook", "raised forecast",
    "dividend increase", "special dividend", "buyback", "share repurchase",
    "acquisition", "merger", "deal", "partnership", "contract win",
    "fda approval", "approval", "approved", "cleared", "authorized",
    "exceed", "exceeds", "exceeded", "top estimate", "above estimate",
    "rally", "rallies", "jump", "jumps", "climb", "climbs",
    "expansion", "growth acceleration", "margin expansion",
    "ai", "artificial intelligence", "innovation", "launched", "launch",
    "robust", "resilient", "momentum", "bullish",
]

MODERATE_POSITIVE = [
    "profit", "profits", "revenue growth", "sales growth", "positive",
    "gains", "gain", "rise", "rises", "recover", "recovery",
    "steady", "stable", "consistent", "confident", "optimistic",
    "opportunities", "opportunity", "promising", "strong demand",
    "market share", "new product", "new service", "expansion plan",
    "dividend", "cash flow", "free cash flow", "margin improvement",
    "cost reduction", "efficiency", "restructuring success",
    "analyst", "price target", "overweight", "outperform rating",
]

MODERATE_NEGATIVE = [
    "miss", "misses", "missed", "below estimate", "disappoint", "disappoints",
    "disappoint", "weak", "slow", "slowdown", "concerns", "concern",
    "challenges", "headwinds", "uncertainty", "volatile", "volatility",
    "competition", "competitive pressure", "margin pressure",
    "cost increase", "inflation impact", "supply chain",
    "downgrade", "downgraded", "reduce", "reduced", "cut target",
    "lowered guidance", "lowered outlook", "revised lower",
    "layoffs", "job cuts", "restructuring", "write-down", "writedown",
    "loss", "losses", "decline", "declines", "fell", "fall",
    "investigation", "probe", "scrutiny", "antitrust",
    "warning", "caution", "risk", "risks",
]

STRONG_NEGATIVE = [
    "crash", "plunge", "plunges", "collapse", "collapses", "tumble", "tumbles",
    "fraud", "scandal", "misconduct", "corruption", "bribery",
    "lawsuit", "sued", "class action", "settlement", "fine", "penalty",
    "bankruptcy", "bankrupt", "default", "insolvency",
    "recall", "safety issue", "safety concern", "accident",
    "ceo resign", "cfo resign", "executive depart", "management change",
    "massive layoff", "plant closure", "shut down",
    "sanctions", "ban", "blocked", "rejected", "denied",
    "profit warning", "guidance cut", "guidance withdrawal",
    "data breach", "cyberattack", "hack",
    "earnings miss", "revenue miss", "sales decline",
    "sell rating", "underperform", "strong sell",
]

# Temi delle notizie
THEMES = {
    "Earnings":       ["earnings", "revenue", "profit", "sales", "quarter", "fy", "eps", "guidance"],
    "M&A":            ["acquisition", "merger", "takeover", "buyout", "deal", "acquire", "bid"],
    "Prodotto/Inno.": ["product", "launch", "innovation", "technology", "ai", "patent", "fda", "approval"],
    "Regolatori":     ["regulation", "regulatory", "antitrust", "sec", "investigation", "lawsuit", "fine"],
    "Management":     ["ceo", "cfo", "executive", "leadership", "board", "resign", "appoint"],
    "Macro":          ["interest rate", "inflation", "recession", "gdp", "trade", "tariff", "geopolit"],
    "ESG":            ["sustainability", "esg", "climate", "carbon", "renewable", "social"],
    "Dividendi":      ["dividend", "buyback", "repurchase", "payout", "yield"],
}


class NewsAnalyzer:
    """
    Analizza le notizie recenti di ogni titolo selezionato.
    Usa yfinance per scaricare i titoli degli articoli (gratuito).
    """

    NEWS_DAYS = 14        # Finestra notizie: ultime 2 settimane
    MAX_NEWS  = 20        # Max notizie per ticker
    CACHE_TTL = 6         # Ore prima di aggiornare la cache news

    def __init__(self, ticker_info: Dict):
        self.info  = ticker_info
        self._cache: Dict[str, dict] = {}

    # ─── METODO PRINCIPALE ───────────────────────────────────────

    def analyze_tickers(self, tickers: List[str]) -> Dict[str, dict]:
        """
        Analizza le notizie per una lista di ticker.

        Returns:
            Dict {ticker: {
                news_score: 0-100,
                signal: 'POSITIVO'|'NEUTRO'|'NEGATIVO',
                sentiment_label: 'Molto positivo'|...,
                headlines: [str],
                themes: [str],
                article_count: int,
                latest_date: str,
                summary: str,
            }}
        """
        results = {}
        total = len(tickers)

        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance non disponibile per news — sentiment neutro su tutti i titoli")
            return {t: self._neutral_result() for t in tickers}

        print(f"   Analisi notizie per {total} titoli...")

        for i, ticker in enumerate(tickers):
            try:
                result = self._analyze_single(ticker, yf)
                results[ticker] = result
                signal_icon = "🟢" if result["signal"] == "POSITIVO" else (
                              "🔴" if result["signal"] == "NEGATIVO" else "⚪")
                if (i + 1) % 5 == 0 or i == total - 1:
                    print(f"   News: {i+1}/{total} titoli analizzati")
                logger.debug(f"News {ticker}: score={result['news_score']:.0f} {signal_icon} "
                             f"({result['article_count']} articoli)")
                # Pausa per non sovraccaricare yfinance
                time.sleep(0.3)
            except Exception as e:
                logger.warning(f"Errore news {ticker}: {e}")
                results[ticker] = self._neutral_result()

        return results

    # ─── ANALISI SINGOLO TICKER ──────────────────────────────────

    def _analyze_single(self, ticker: str, yf) -> dict:
        """Scarica e analizza le notizie di un singolo ticker."""
        try:
            tk   = yf.Ticker(ticker)
            news = tk.news or []
        except Exception:
            return self._neutral_result()

        if not news:
            return self._neutral_result()

        # Filtra le notizie degli ultimi N giorni
        cutoff    = datetime.now() - timedelta(days=self.NEWS_DAYS)
        headlines = []
        raw_texts = []

        for article in news[:self.MAX_NEWS]:
            # yfinance restituisce timestamp Unix
            ts = article.get("providerPublishTime", 0)
            if ts:
                pub_date = datetime.fromtimestamp(ts)
                if pub_date < cutoff:
                    continue

            title   = article.get("title", "")
            summary = article.get("summary", "") or ""
            if title:
                headlines.append(title)
                raw_texts.append(f"{title} {summary}".lower())

        if not headlines:
            return self._neutral_result()

        # Sentiment scoring
        pos_strong, pos_mod, neg_mod, neg_strong = 0, 0, 0, 0
        for text in raw_texts:
            for w in STRONG_POSITIVE:
                if w in text: pos_strong += 1
            for w in MODERATE_POSITIVE:
                if w in text: pos_mod += 1
            for w in MODERATE_NEGATIVE:
                if w in text: neg_mod += 1
            for w in STRONG_NEGATIVE:
                if w in text: neg_strong += 1

        # Score grezzo: ponderato per intensità
        raw_score = (pos_strong * 3 + pos_mod * 1 - neg_mod * 1.5 - neg_strong * 4)
        n         = len(headlines)

        # Normalizza in 0-100 (50 = neutro)
        # Clip a ±(n * 3) per non essere troppo estremi
        max_range = max(n * 3, 1)
        normalized = (raw_score / max_range) * 40 + 50  # 50 ± 40
        news_score = float(max(10, min(90, normalized)))

        # Segnale
        if news_score >= 62:
            signal = "POSITIVO"
            label  = "Molto positivo" if news_score >= 75 else "Positivo"
        elif news_score <= 38:
            signal = "NEGATIVO"
            label  = "Molto negativo" if news_score <= 25 else "Negativo"
        else:
            signal = "NEUTRO"
            label  = "Neutro"

        # Temi identificati
        all_text = " ".join(raw_texts)
        themes   = [t for t, kws in THEMES.items()
                    if any(kw in all_text for kw in kws)]

        # Data ultima notizia
        latest_ts = max((a.get("providerPublishTime", 0) for a in news[:5]), default=0)
        latest_date = datetime.fromtimestamp(latest_ts).strftime("%d/%m/%Y") if latest_ts else "—"

        # Breve sommario testuale
        summary_txt = self._make_summary(news_score, signal, themes, n, headlines)

        return {
            "news_score":      round(news_score, 1),
            "signal":          signal,
            "sentiment_label": label,
            "headlines":       headlines[:6],           # prime 6 per il report
            "themes":          themes,
            "article_count":   n,
            "latest_date":     latest_date,
            "summary":         summary_txt,
            "pos_signals":     pos_strong + pos_mod,
            "neg_signals":     neg_strong + neg_mod,
        }

    # ─── HELPERS ─────────────────────────────────────────────────

    def _make_summary(self, score: float, signal: str,
                      themes: List[str], n: int, headlines: List[str]) -> str:
        """Genera un breve testo descrittivo del sentiment."""
        if signal == "POSITIVO":
            intro = f"Sentiment positivo su {n} articoli recenti."
        elif signal == "NEGATIVO":
            intro = f"Sentiment negativo su {n} articoli recenti — attenzione."
        else:
            intro = f"Notizie miste/neutre ({n} articoli)."

        if themes:
            intro += f" Temi: {', '.join(themes[:3])}."

        return intro

    def _neutral_result(self) -> dict:
        """Risultato neutro quando non ci sono notizie o si verifica un errore."""
        return {
            "news_score":      50.0,
            "signal":          "NEUTRO",
            "sentiment_label": "Nessuna notizia recente",
            "headlines":       [],
            "themes":          [],
            "article_count":   0,
            "latest_date":     "—",
            "summary":         "Nessuna notizia recente disponibile.",
            "pos_signals":     0,
            "neg_signals":     0,
        }
