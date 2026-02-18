"""
Modulo per il download e la cache dei dati di mercato.
Usa yfinance con sistema di cache locale per evitare download ripetuti.
"""

import os
import json
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger("portfolio_optimizer")


class DataDownloader:
    """
    Scarica e gestisce i dati storici e fondamentali.
    
    La cache locale evita download ripetuti nella stessa sessione o
    entro il periodo di validità configurato (default 24h).
    """

    CACHE_DIR = "data/cache"
    CACHE_EXPIRY_HOURS = 24

    def __init__(self, period: str = "10y", refresh: bool = False):
        """
        Args:
            period: Periodo storico (es. '5y', '10y')
            refresh: Se True, ignora la cache e riscarica tutto
        """
        self.period = period
        self.refresh = refresh
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs("data", exist_ok=True)

    # ─── METODI PUBBLICI ─────────────────────────────────────────

    def download_all(self, tickers: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Scarica prezzi storici e informazioni fondamentali per tutti i ticker.
        
        Returns:
            price_data: DataFrame con colonne = ticker, index = date
            ticker_info: Dict {ticker: {info fondamentali}}
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance non installato. Esegui: pip install yfinance"
            )

        print(f"   Scaricando dati per {len(tickers)} ticker...")
        price_data = self._download_prices(tickers, yf)
        ticker_info = self._download_fundamentals(
            [t for t in tickers if t in price_data.columns], yf
        )
        return price_data, ticker_info

    # ─── PREZZI STORICI ──────────────────────────────────────────

    def _download_prices(self, tickers: List[str], yf) -> pd.DataFrame:
        """Scarica i prezzi di chiusura aggiustati."""
        cache_file = os.path.join(self.CACHE_DIR, f"prices_{self.period}.pkl")

        if not self.refresh and self._is_cache_valid(cache_file):
            logger.info("Caricamento prezzi dalla cache")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        logger.info(f"Download prezzi per {len(tickers)} ticker, periodo {self.period}")

        # Download a batch per efficienza
        data = pd.DataFrame()
        batch_size = 50
        failed = []

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            try:
                raw = yf.download(
                    batch,
                    period=self.period,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    timeout=30
                )

                if isinstance(raw.columns, pd.MultiIndex):
                    # yfinance restituisce MultiIndex (metric, ticker)
                    if "Close" in raw.columns.get_level_values(0):
                        closes = raw["Close"].copy()
                    else:
                        closes = raw.iloc[:, ::5].copy()
                    # Assicura che le colonne siano i ticker (non tuple)
                    closes.columns = [
                        str(c[1]) if isinstance(c, tuple) else str(c)
                        for c in closes.columns
                    ]
                else:
                    if "Close" in raw.columns:
                        closes = raw[["Close"]].copy()
                        if len(batch) == 1:
                            closes = pd.DataFrame({batch[0]: closes.squeeze()})
                    else:
                        closes = raw.copy()

                # Singolo ticker senza MultiIndex
                if len(batch) == 1 and not isinstance(raw.columns, pd.MultiIndex):
                    closes = pd.DataFrame({batch[0]: closes.squeeze()})

                # Assicura che ogni colonna sia 1D (nessuna Series nested)
                for col in closes.columns:
                    if isinstance(closes[col], pd.DataFrame):
                        closes[col] = closes[col].iloc[:, 0]

                data = pd.concat([data, closes], axis=1)

            except Exception as e:
                logger.warning(f"Errore batch {batch[:3]}...: {e}")
                failed.extend(batch)

        # Rimuovi colonne quasi vuote (< 2 anni di dati)
        min_points = 252 * 2
        data = data.dropna(thresh=min_points, axis=1)
        data = data.sort_index()

        # Salva cache
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Prezzi scaricati: {data.shape[1]} ticker validi")
        return data

    # ─── DATI FONDAMENTALI ───────────────────────────────────────

    def _download_fundamentals(self, tickers: List[str], yf) -> Dict:
        """Scarica i dati fondamentali da yfinance."""
        cache_file = os.path.join(self.CACHE_DIR, "fundamentals.pkl")

        if not self.refresh and self._is_cache_valid(cache_file):
            logger.info("Caricamento fondamentali dalla cache")
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            # Aggiorna solo i ticker mancanti
            missing = [t for t in tickers if t not in cached]
            if not missing:
                return cached
            tickers_to_download = missing
            existing = cached
        else:
            tickers_to_download = tickers
            existing = {}

        all_info = dict(existing)
        print(f"   Scaricando fondamentali per {len(tickers_to_download)} ticker...")

        for i, ticker in enumerate(tickers_to_download):
            try:
                tk = yf.Ticker(ticker)
                info = tk.info

                # Estrai i campi rilevanti
                # Per ticker APAC/EM yfinance non sempre popola country:
                # usiamo il mapping statico come fallback
                from utils.config import Config as _Cfg
                country_raw = info.get("country", "Unknown") or "Unknown"
                country = _Cfg.TICKER_COUNTRY_MAP.get(ticker, country_raw)
                all_info[ticker] = {
                    "name": info.get("longName", ticker),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "country": country,
                    "currency": info.get("currency", "USD"),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
                    "forward_pe": info.get("forwardPE"),
                    "pb_ratio": info.get("priceToBook"),
                    "ps_ratio": info.get("priceToSalesTrailing12Months"),
                    "roe": info.get("returnOnEquity"),
                    "roa": info.get("returnOnAssets"),
                    "profit_margin": info.get("profitMargins"),
                    "gross_margin": info.get("grossMargins"),
                    "operating_margin": info.get("operatingMargins"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "earnings_growth": info.get("earningsGrowth"),
                    "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
                    "free_cashflow": info.get("freeCashflow"),
                    "fcf_yield": self._calc_fcf_yield(info),
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "quick_ratio": info.get("quickRatio"),
                    "dividend_yield": info.get("dividendYield"),
                    "payout_ratio": info.get("payoutRatio"),
                    "beta": info.get("beta"),
                    "52w_high": info.get("fiftyTwoWeekHigh"),
                    "52w_low": info.get("fiftyTwoWeekLow"),
                    "analyst_target": info.get("targetMeanPrice"),
                    "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "employees": info.get("fullTimeEmployees"),
                    "description": info.get("longBusinessSummary", "")[:300],
                }

                if (i + 1) % 10 == 0:
                    print(f"   Fondamentali: {i+1}/{len(tickers_to_download)}")

            except Exception as e:
                logger.warning(f"Errore fondamentali {ticker}: {e}")
                all_info[ticker] = self._empty_info(ticker)

        # Salva cache aggiornata
        with open(cache_file, "wb") as f:
            pickle.dump(all_info, f)

        return all_info

    # ─── UTILITY ─────────────────────────────────────────────────

    def _is_cache_valid(self, cache_file: str) -> bool:
        """Controlla se la cache è recente."""
        if not os.path.exists(cache_file):
            return False
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        return datetime.now() - mtime < timedelta(hours=self.CACHE_EXPIRY_HOURS)

    def _calc_fcf_yield(self, info: dict) -> Optional[float]:
        """Calcola FCF Yield = Free Cash Flow / Market Cap."""
        fcf = info.get("freeCashflow")
        mc = info.get("marketCap")
        if fcf and mc and mc > 0:
            return fcf / mc
        return None

    def _empty_info(self, ticker: str) -> dict:
        """Info vuote per ticker con errore."""
        return {
            "name": ticker, "sector": "Unknown", "industry": "Unknown",
            "country": "Unknown", "currency": "USD",
            **{k: None for k in [
                "market_cap", "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio",
                "roe", "roa", "profit_margin", "gross_margin", "operating_margin",
                "revenue_growth", "earnings_growth", "earnings_quarterly_growth",
                "free_cashflow", "fcf_yield", "debt_to_equity", "current_ratio",
                "quick_ratio", "dividend_yield", "payout_ratio", "beta",
                "52w_high", "52w_low", "analyst_target", "current_price",
                "employees", "description"
            ]}
        }
