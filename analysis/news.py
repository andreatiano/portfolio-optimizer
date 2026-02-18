"""
Modulo di analisi delle notizie recenti per ogni titolo selezionato.

Funzionamento:
  1. Scarica le ultime notizie da yfinance (gratis, nessuna API key)
  2. Calcola un sentiment score con analisi keyword (no dipendenze extra)
  3. Produce un news_score 0-100 e un segnale: POSITIVO / NEUTRO / NEGATIVO

NOTA: Compatibile con la nuova struttura yfinance (dati dentro 'content').
"""

import logging
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
    "weak", "slow", "slowdown", "concerns", "concern",
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

THEMES = {
    "Earnings":       ["earnings", "revenue", "profit", "sales", "quarter", "eps", "guidance"],
    "M&A":            ["acquisition", "merger", "takeover", "buyout", "deal", "acquire", "bid"],
    "Prodotto/Inno.": ["product", "launch", "innovation", "technology", "ai", "patent", "fda", "approval"],
    "Regolatori":     ["regulation", "regulatory", "antitrust", "sec", "investigation", "lawsuit", "fine"],
    "Management":     ["ceo", "cfo", "executive", "leadership", "board", "resign", "appoint"],
    "Macro":          ["interest rate", "inflation", "recession", "gdp", "trade", "tariff"],
    "Dividendi":      ["dividend", "buyback", "repurchase", "payout", "yield"],
}


class NewsAnalyzer:
    """
    Analizza le notizie recenti di ogni titolo selezionato.
    Compatibile con vecchia e nuova struttura dati di yfinance.
    """

    NEWS_DAYS = 14
    MAX_NEWS  = 20

    def __init__(self, ticker_info: Dict):
        self.info = ticker_info

    # ─── METODO PRINCIPALE ───────────────────────────────────────

    def analyze_tickers(self, tickers: List[str]) -> Dict[str, dict]:
        results = {}
        total   = len(tickers)

        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance non disponibile — sentiment neutro su tutti i titoli")
            return {t: self._neutral_result() for t in tickers}

        print(f"   Analisi notizie per {total} titoli...")

        for i, ticker in enumerate(tickers):
            try:
                result = self._analyze_single(ticker, yf)
                results[ticker] = result
                if (i + 1) % 5 == 0 or i == total - 1:
                    print(f"   News: {i+1}/{total} analizzati")
                time.sleep(0.3)
            except Exception as e:
                logger.warning(f"Errore news {ticker}: {e}")
                results[ticker] = self._neutral_result()

        return results

    # ─── ANALISI SINGOLO TICKER ──────────────────────────────────

    def _analyze_single(self, ticker: str, yf) -> dict:
        try:
            tk   = yf.Ticker(ticker)
            news = tk.news or []
        except Exception:
            return self._neutral_result()

        if not news:
            return self._neutral_result()

        cutoff    = datetime.now() - timedelta(days=self.NEWS_DAYS)
        headlines = []
        raw_texts = []

        for article in news[:self.MAX_NEWS]:
            # ── Nuova struttura: tutto dentro 'content' ──────────
            content = article.get("content", {})

            if content:
                title   = content.get("title", "")
                summary = content.get("summary", "") or content.get("description", "") or ""
                # Data: prova pubDate (stringa ISO) oppure displayTime
                pub_str = content.get("pubDate", "") or content.get("displayTime", "")
                ts = 0
                if pub_str:
                    try:
                        ts = int(datetime.fromisoformat(
                            pub_str.replace("Z", "+00:00")).timestamp())
                    except Exception:
                        ts = 0
            else:
                # ── Vecchia struttura: campi piatti ─────────────
                title   = article.get("title", "")
                summary = article.get("summary", "") or ""
                ts      = article.get("providerPublishTime", 0)

            # Filtra per data
            if ts:
                try:
                    pub_date = datetime.fromtimestamp(ts)
                    if pub_date < cutoff:
                        continue
                except Exception:
                    pass

            if title:
                headlines.append(title)
                raw_texts.append(f"{title} {summary}".lower())

        if not headlines:
            return self._neutral_result()

        # ── Sentiment scoring ────────────────────────────────────
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

        raw_score = (pos_strong * 3 + pos_mod * 1 - neg_mod * 1.5 - neg_strong * 4)
        n         = len(headlines)
        max_range = max(n * 3, 1)
        normalized = (raw_score / max_range) * 40 + 50
        news_score = float(max(10, min(90, normalized)))

        # ── Segnale ──────────────────────────────────────────────
        if news_score >= 62:
            signal = "POSITIVO"
            label  = "Molto positivo" if news_score >= 75 else "Positivo"
        elif news_score <= 38:
            signal = "NEGATIVO"
            label  = "Molto negativo" if news_score <= 25 else "Negativo"
        else:
            signal = "NEUTRO"
            label  = "Neutro"

        # ── Temi ─────────────────────────────────────────────────
        all_text = " ".join(raw_texts)
        themes   = [t for t, kws in THEMES.items()
                    if any(kw in all_text for kw in kws)]

        # ── Data ultima notizia ──────────────────────────────────
        latest_date = "—"
        for art in news[:5]:
            c = art.get("content", {})
            pub_str = c.get("pubDate", "") if c else art.get("providerPublishTime", 0)
            if isinstance(pub_str, str) and pub_str:
                try:
                    dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                    latest_date = dt.strftime("%d/%m/%Y")
                    break
                except Exception:
                    pass
            elif isinstance(pub_str, (int, float)) and pub_str:
                try:
                    latest_date = datetime.fromtimestamp(pub_str).strftime("%d/%m/%Y")
                    break
                except Exception:
                    pass

        summary_txt = self._make_summary(signal, themes, n)

        return {
            "news_score":      round(news_score, 1),
            "signal":          signal,
            "sentiment_label": label,
            "headlines":       headlines[:6],
            "themes":          themes,
            "article_count":   n,
            "latest_date":     latest_date,
            "summary":         summary_txt,
            "pos_signals":     pos_strong + pos_mod,
            "neg_signals":     neg_strong + neg_mod,
        }

    # ─── HELPERS ─────────────────────────────────────────────────

    def _make_summary(self, signal: str, themes: List[str], n: int) -> str:
        if signal == "POSITIVO":
            intro = f"Sentiment positivo su {n} articoli recenti."
        elif signal == "NEGATIVO":
            intro = f"Sentiment negativo su {n} articoli — attenzione."
        else:
            intro = f"Notizie miste/neutre ({n} articoli)."
        if themes:
            intro += f" Temi: {', '.join(themes[:3])}."
        return intro

    def _neutral_result(self) -> dict:
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
