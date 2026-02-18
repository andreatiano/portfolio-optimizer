"""
demo.py - Demo del Portfolio Optimizer con dati SIMULATI.

Esegui questo script per vedere il funzionamento del sistema
senza dover scaricare dati da internet.

Utilizzo:
    python demo.py
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_synthetic_data(tickers, years=10, seed=42):
    """
    Genera prezzi storici sintetici con caratteristiche realistiche:
    - Trend di crescita diversi per settore
    - Volatilità variabile
    - Correlazioni intra-settore
    """
    np.random.seed(seed)
    n_days = years * 252
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="B")

    # Parametri per settore (drift annuo, volatilità annua)
    sector_params = {
        "AAPL": (0.22, 0.25), "MSFT": (0.20, 0.22), "GOOGL": (0.18, 0.28),
        "NVDA": (0.35, 0.50), "META": (0.15, 0.35), "V": (0.16, 0.18),
        "MA": (0.17, 0.19), "JNJ": (0.08, 0.13), "UNH": (0.18, 0.20),
        "PG": (0.07, 0.12), "KO": (0.06, 0.11), "WMT": (0.10, 0.16),
        "JPM": (0.12, 0.22), "BRK-B": (0.11, 0.18), "HON": (0.10, 0.19),
        "COST": (0.19, 0.22), "NEE": (0.09, 0.16), "AMZN": (0.22, 0.32),
        "ASML.AS": (0.25, 0.30), "SAP.DE": (0.12, 0.22),
        "NESN.SW": (0.07, 0.12), "NOVN.SW": (0.10, 0.15),
        "MC.PA": (0.15, 0.25), "OR.PA": (0.11, 0.16), "AZN.L": (0.13, 0.18),
    }

    prices = {}
    for ticker in tickers:
        mu, sigma = sector_params.get(ticker, (0.10, 0.22))
        dt = 1 / 252
        # Geometric Brownian Motion
        daily_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(n_days)
        price = 100 * np.exp(np.cumsum(daily_ret))
        prices[ticker] = price

    return pd.DataFrame(prices, index=dates)


def generate_synthetic_info(tickers):
    """Genera dati fondamentali sintetici realistici per universo globale."""
    # Settore per ticker
    SECTOR_MAP = {
        # USA Tech
        "AAPL":"Technology","MSFT":"Technology","GOOGL":"Technology",
        "NVDA":"Technology","META":"Communication Services","ADBE":"Technology",
        "ORCL":"Technology","QCOM":"Technology","TXN":"Technology",
        # USA Healthcare
        "JNJ":"Healthcare","UNH":"Healthcare","LLY":"Healthcare",
        "MRK":"Healthcare","ISRG":"Healthcare","TMO":"Healthcare",
        # USA Finance
        "V":"Financial Services","MA":"Financial Services","JPM":"Financial Services",
        "BRK-B":"Financial Services","BLK":"Financial Services","GS":"Financial Services",
        # USA Consumer
        "AMZN":"Consumer Cyclical","COST":"Consumer Defensive","WMT":"Consumer Defensive",
        "MCD":"Consumer Cyclical","PG":"Consumer Defensive","KO":"Consumer Defensive",
        "PEP":"Consumer Defensive","NKE":"Consumer Cyclical",
        # USA Industrials/Utilities
        "HON":"Industrials","CAT":"Industrials","NEE":"Utilities",
        "AMT":"Real Estate","XOM":"Energy","CVX":"Energy",
        # Europa Tech/Industrial
        "ASML.AS":"Technology","SAP.DE":"Technology","SIE.DE":"Industrials",
        "ABB":"Industrials","ABBN.SW":"Industrials",
        # Europa Healthcare
        "NESN.SW":"Consumer Defensive","NOVN.SW":"Healthcare",
        "AZN.L":"Healthcare","GSK.L":"Healthcare","ROG.SW":"Healthcare",
        # Europa Finance/Consumer
        "MC.PA":"Consumer Cyclical","OR.PA":"Consumer Defensive",
        "ULVR.L":"Consumer Defensive","ALV.DE":"Financial Services",
        "HSBA.L":"Financial Services","BNP.PA":"Financial Services",
        "ITX.MC":"Consumer Cyclical","AIR.PA":"Industrials",
        # Giappone
        "7203.T":"Consumer Cyclical","6758.T":"Technology",
        "9984.T":"Communication Services","6861.T":"Technology",
        "8306.T":"Financial Services","4519.T":"Healthcare",
        # Corea
        "005930.KS":"Technology","000660.KS":"Technology","051910.KS":"Basic Materials",
        # Australia
        "BHP.AX":"Basic Materials","CSL.AX":"Healthcare",
        "CBA.AX":"Financial Services","WES.AX":"Consumer Defensive",
        # Asia (HK/Singapore/India)
        "0700.HK":"Communication Services","9988.HK":"Consumer Cyclical",
        "2318.HK":"Financial Services","D05.SI":"Financial Services",
        "TCS.NS":"Technology","INFY.NS":"Technology","HDFCBANK.NS":"Financial Services",
        "RELIANCE.NS":"Energy",
        # Canada
        "SHOP.TO":"Technology","RY.TO":"Financial Services",
        "TD.TO":"Financial Services","CNR.TO":"Industrials","ENB.TO":"Energy",
        # Brasile
        "VALE3.SA":"Basic Materials","PETR4.SA":"Energy",
        "ITUB4.SA":"Financial Services","WEGE3.SA":"Industrials",
    }

    # Paese per ticker (usato per la mappa geografica nel report)
    COUNTRY_MAP = {
        # USA: nessun suffisso
        "7203.T":"Japan","6758.T":"Japan","9984.T":"Japan","6861.T":"Japan",
        "8306.T":"Japan","4519.T":"Japan","9433.T":"Japan","6501.T":"Japan",
        "005930.KS":"South Korea","000660.KS":"South Korea","051910.KS":"South Korea",
        "BHP.AX":"Australia","CSL.AX":"Australia","CBA.AX":"Australia","WES.AX":"Australia",
        "0700.HK":"China","9988.HK":"China","2318.HK":"China","0941.HK":"China",
        "D05.SI":"Singapore","O39.SI":"Singapore","Z74.SI":"Singapore",
        "RELIANCE.NS":"India","TCS.NS":"India","INFY.NS":"India",
        "HDFCBANK.NS":"India","WIPRO.NS":"India",
        "SHOP.TO":"Canada","RY.TO":"Canada","TD.TO":"Canada",
        "CNR.TO":"Canada","ENB.TO":"Canada","BAM.TO":"Canada",
        "VALE3.SA":"Brazil","PETR4.SA":"Brazil","ITUB4.SA":"Brazil",
        "ABEV3.SA":"Brazil","WEGE3.SA":"Brazil",
        "SAP.DE":"Germany","SIE.DE":"Germany","ALV.DE":"Germany",
        "MRK.DE":"Germany","ADS.DE":"Germany","BMW.DE":"Germany",
        "MC.PA":"France","OR.PA":"France","TTE.PA":"France",
        "SAN.PA":"France","AIR.PA":"France","BNP.PA":"France",
        "ULVR.L":"United Kingdom","HSBA.L":"United Kingdom","GSK.L":"United Kingdom",
        "AZN.L":"United Kingdom","SHEL.L":"United Kingdom","RIO.L":"United Kingdom",
        "NESN.SW":"Switzerland","NOVN.SW":"Switzerland","ROG.SW":"Switzerland","ABBN.SW":"Switzerland",
        "ASML.AS":"Netherlands","HEIA.AS":"Netherlands","INGA.AS":"Netherlands",
        "ENI.MI":"Italy","ENEL.MI":"Italy","ISP.MI":"Italy",
        "ITX.MC":"Spain","SAN.MC":"Spain","IBE.MC":"Spain",
        "NOVO-B.CO":"Denmark",
    }

    np.random.seed(123)
    info = {}
    for ticker in tickers:
        sector = SECTOR_MAP.get(ticker, "Industrials")
        country = COUNTRY_MAP.get(ticker, "US")
        # Valuta in base al paese
        if country == "US" or "." not in ticker:
            currency = "USD"
        elif country in ("United Kingdom",):
            currency = "GBP"
        elif country in ("Switzerland",):
            currency = "CHF"
        elif country in ("Japan",):
            currency = "JPY"
        elif country in ("South Korea",):
            currency = "KRW"
        elif country in ("Australia",):
            currency = "AUD"
        elif country in ("Canada",):
            currency = "CAD"
        elif country in ("Brazil",):
            currency = "BRL"
        elif country in ("India",):
            currency = "INR"
        elif country in ("China", "Hong Kong", "Singapore"):
            currency = "HKD"
        else:
            currency = "EUR"

        # Fondamentali realistici per settore
        if sector == "Technology":
            pe, roe, margin = np.random.uniform(20,45), np.random.uniform(0.15,0.50), np.random.uniform(0.18,0.35)
            rev_growth, earn_growth = np.random.uniform(0.08,0.25), np.random.uniform(0.10,0.30)
        elif sector == "Healthcare":
            pe, roe, margin = np.random.uniform(18,35), np.random.uniform(0.12,0.30), np.random.uniform(0.12,0.25)
            rev_growth, earn_growth = np.random.uniform(0.05,0.15), np.random.uniform(0.08,0.20)
        elif sector == "Consumer Defensive":
            pe, roe, margin = np.random.uniform(16,28), np.random.uniform(0.10,0.25), np.random.uniform(0.06,0.18)
            rev_growth, earn_growth = np.random.uniform(0.03,0.12), np.random.uniform(0.05,0.12)
        elif sector == "Financial Services":
            pe, roe, margin = np.random.uniform(10,20), np.random.uniform(0.10,0.20), np.random.uniform(0.20,0.35)
            rev_growth, earn_growth = np.random.uniform(0.04,0.12), np.random.uniform(0.05,0.15)
        elif sector == "Communication Services":
            pe, roe, margin = np.random.uniform(18,35), np.random.uniform(0.12,0.28), np.random.uniform(0.15,0.30)
            rev_growth, earn_growth = np.random.uniform(0.06,0.18), np.random.uniform(0.08,0.20)
        elif sector == "Basic Materials":
            pe, roe, margin = np.random.uniform(10,22), np.random.uniform(0.08,0.20), np.random.uniform(0.10,0.25)
            rev_growth, earn_growth = np.random.uniform(0.02,0.12), np.random.uniform(0.03,0.15)
        else:
            pe, roe, margin = np.random.uniform(14,30), np.random.uniform(0.08,0.20), np.random.uniform(0.08,0.20)
            rev_growth, earn_growth = np.random.uniform(0.04,0.15), np.random.uniform(0.05,0.15)

        # Nome leggibile
        name_clean = (ticker
            .replace(".T","").replace(".KS","").replace(".AX","").replace(".HK","")
            .replace(".SI","").replace(".NS","").replace(".TO","").replace(".SA","")
            .replace(".DE","").replace(".PA","").replace(".L","").replace(".SW","")
            .replace(".AS","").replace(".MI","").replace(".MC","").replace(".CO","")
            .replace(".ST",""))
        # Aggiungi paese per chiarezza
        country_short = {
            "Japan":"🇯🇵","South Korea":"🇰🇷","Australia":"🇦🇺","China":"🇨🇳",
            "Singapore":"🇸🇬","India":"🇮🇳","Canada":"🇨🇦","Brazil":"🇧🇷",
            "Germany":"🇩🇪","France":"🇫🇷","United Kingdom":"🇬🇧","Switzerland":"🇨🇭",
            "Netherlands":"🇳🇱","Italy":"🇮🇹","Spain":"🇪🇸","Denmark":"🇩🇰",
        }.get(country, "")

        info[ticker] = {
            "name": f"{name_clean} {country_short}".strip(),
            "sector": sector,
            "industry": sector,
            "country": country,
            "currency": currency,
            "market_cap": np.random.uniform(10e9, 800e9),
            "pe_ratio": pe,
            "forward_pe": pe * 0.9,
            "pb_ratio": np.random.uniform(2, 10),
            "ps_ratio": np.random.uniform(1, 8),
            "roe": roe,
            "roa": roe * 0.5,
            "profit_margin": margin,
            "gross_margin": margin * 2,
            "operating_margin": margin * 1.3,
            "revenue_growth": rev_growth,
            "earnings_growth": earn_growth,
            "earnings_quarterly_growth": earn_growth * 0.8,
            "free_cashflow": np.random.uniform(1e9, 50e9),
            "fcf_yield": np.random.uniform(0.01, 0.06),
            "debt_to_equity": np.random.uniform(10, 150),
            "current_ratio": np.random.uniform(1.0, 3.0),
            "quick_ratio": np.random.uniform(0.8, 2.5),
            "dividend_yield": np.random.uniform(0, 0.04),
            "payout_ratio": np.random.uniform(0, 0.5),
            "beta": np.random.uniform(0.5, 1.8),
            "52w_high": None, "52w_low": None,
            "analyst_target": None, "current_price": None,
            "employees": int(np.random.uniform(5000, 300000)),
            "description": f"Azienda leader nel settore {sector} ({country}).",
        }
    return info


def generate_synthetic_news(tickers, ticker_info):
    """
    Genera notizie sintetiche realistiche per la demo.
    In produzione (main.py) vengono usate le notizie reali da yfinance.

    Logica:
    - Titoli tech growth → tendenza positiva (AI, earnings beat)
    - Titoli energy/materiali → più volatili (prezzi commodity)
    - Titoli finance → mixed (tassi, regolatori)
    - Occasionalmente genera notizie negative per testare il filtro
    """
    import random
    rng = random.Random(99)

    POSITIVE_TEMPLATES = [
        "{name} beats Q4 earnings estimates, raises full-year guidance",
        "{name} announces record revenue growth driven by AI products",
        "{name} expands into new markets, strong demand outlook",
        "{name} upgraded to Buy by major analyst, price target raised",
        "{name} announces $5B share buyback program",
        "{name} wins major contract, accelerating growth strategy",
        "{name} reports strong free cash flow, dividend increase likely",
        "{name} new product launch exceeds early sales expectations",
        "{name} partnership with leading tech firm boosts outlook",
        "{name} margin expansion continues, efficiency gains recognized",
    ]
    NEUTRAL_TEMPLATES = [
        "{name} reports results in line with estimates",
        "{name} maintains full-year guidance amid macro uncertainty",
        "{name} investor day highlights long-term strategy",
        "{name} CFO comments on stable demand environment",
        "{name} analyst day: steady growth trajectory confirmed",
        "{name} announces leadership transition, operations unaffected",
    ]
    NEGATIVE_TEMPLATES = [
        "{name} misses quarterly revenue estimates, shares fall",
        "{name} faces regulatory scrutiny in key markets",
        "{name} lowers guidance citing macro headwinds",
        "{name} CEO departure raises investor concerns",
        "{name} margin pressure intensifies, cost cuts announced",
        "{name} faces increased competition, market share at risk",
    ]

    # Temi per settore
    SECTOR_THEMES = {
        "Technology":             ["Earnings", "Prodotto/Inno.", "M&A"],
        "Healthcare":             ["Earnings", "Prodotto/Inno.", "Regolatori"],
        "Financial Services":     ["Earnings", "Regolatori", "Macro"],
        "Consumer Cyclical":      ["Earnings", "Macro", "Prodotto/Inno."],
        "Consumer Defensive":     ["Earnings", "Dividendi", "Macro"],
        "Industrials":            ["Earnings", "M&A", "Macro"],
        "Communication Services": ["Earnings", "Prodotto/Inno.", "Regolatori"],
        "Energy":                 ["Earnings", "Macro", "ESG"],
        "Basic Materials":        ["Earnings", "Macro", "ESG"],
        "Utilities":              ["Earnings", "Dividendi", "Regolatori"],
        "Real Estate":            ["Earnings", "Macro", "Dividendi"],
    }

    results = {}
    for ticker in tickers:
        info   = ticker_info.get(ticker, {})
        name   = (info.get("name", ticker) or ticker).split(" ")[0]
        sector = info.get("sector", "Industrials")

        # Probabilità di sentiment per settore
        if sector == "Technology":
            weights = [0.55, 0.30, 0.15]   # pos, neu, neg
        elif sector in ("Healthcare", "Consumer Defensive"):
            weights = [0.45, 0.35, 0.20]
        elif sector == "Energy":
            weights = [0.35, 0.35, 0.30]
        elif sector == "Financial Services":
            weights = [0.40, 0.35, 0.25]
        else:
            weights = [0.40, 0.35, 0.25]

        choice = rng.choices(["POSITIVO", "NEUTRO", "NEGATIVO"], weights=weights)[0]

        if choice == "POSITIVO":
            templates   = POSITIVE_TEMPLATES
            score_base  = rng.uniform(62, 88)
            label       = "Positivo" if score_base < 75 else "Molto positivo"
        elif choice == "NEGATIVO":
            templates   = NEGATIVE_TEMPLATES
            score_base  = rng.uniform(18, 42)
            label       = "Negativo" if score_base > 28 else "Molto negativo"
        else:
            templates   = NEUTRAL_TEMPLATES
            score_base  = rng.uniform(44, 58)
            label       = "Neutro"

        # Genera 3-6 titoli notizie
        n_articles = rng.randint(3, 6)
        headlines  = [t.format(name=name) for t in rng.choices(templates, k=n_articles)]
        themes     = SECTOR_THEMES.get(sector, ["Earnings", "Macro"])[:rng.randint(1, 3)]

        from datetime import datetime, timedelta
        days_ago   = rng.randint(1, 12)
        date_str   = (datetime.now() - timedelta(days=days_ago)).strftime("%d/%m/%Y")

        raw_score_val = int(score_base * 2 - 100)  # converti 0..100 → -100..+100
        results[ticker] = {
            # Campi usati dal nuovo NewsSentimentAnalyzer
            "news_score":       round(score_base, 1),    # 0..100 normalizzato
            "news_raw_score":   raw_score_val,           # -100..+100
            "signal":           choice,
            "news_signal":      choice,
            "sentiment":        label,
            "news_sentiment":   label,
            "article_count":    n_articles,
            "news_articles":    n_articles,
            "summary":          f"[DEMO] Sentiment {label.lower()} su {n_articles} articoli recenti.",
            "news_summary":     f"[DEMO] Sentiment {label.lower()} su {n_articles} articoli recenti.",
            "key_points":       headlines[:3],
            "news_key_points":  headlines[:3],
            "risk_flags":       ([f"Pressione competitiva in {sector}"] if choice == "NEGATIVO" else []),
            "news_risk_flags":  ([f"Pressione competitiva in {sector}"] if choice == "NEGATIVO" else []),
            "opportunity":      ([f"Crescita {sector} attesa"] if choice == "POSITIVO" else []),
            "news_opportunity": ([f"Crescita {sector} attesa"] if choice == "POSITIVO" else []),
            "timing":           ("Momento favorevole per posizionarsi" if choice == "POSITIVO"
                                 else "Attendere chiarimento prima di entrare" if choice == "NEGATIVO"
                                 else "Monitorare sviluppi nelle prossime settimane"),
            "news_timing":      ("Momento favorevole per posizionarsi" if choice == "POSITIVO"
                                 else "Attendere chiarimento prima di entrare" if choice == "NEGATIVO"
                                 else "Monitorare sviluppi nelle prossime settimane"),
            "favorable":        (True if choice == "POSITIVO" else False if choice == "NEGATIVO" else None),
            "news_favorable":   (True if choice == "POSITIVO" else False if choice == "NEGATIVO" else None),
            "cached":           False,
        }

    return results


def run_demo():
    from analysis.technical import TechnicalAnalyzer
    from analysis.fundamentals import FundamentalAnalyzer
    from analysis.news_sentiment import NewsSentimentAnalyzer
    from analysis.scorer import StockScorer
    from optimization.portfolio import PortfolioOptimizer
    from output.report import ReportGenerator
    from utils.logger import setup_logger

    logger = setup_logger()
    CAPITAL = 15000

    DEMO_TICKERS = [
        # USA (20)
        "AAPL", "MSFT", "GOOGL", "NVDA", "META",
        "V", "MA", "JNJ", "UNH", "LLY",
        "PG", "KO", "WMT", "JPM", "BRK-B",
        "HON", "COST", "NEE", "AMZN", "ADBE",
        # Europa (15)
        "ASML.AS", "SAP.DE", "SIE.DE", "ALV.DE",
        "NESN.SW", "NOVN.SW", "ROG.SW",
        "MC.PA", "OR.PA", "AIR.PA", "BNP.PA",
        "AZN.L", "ULVR.L", "ITX.MC", "ENI.MI",
        # Giappone (5)
        "7203.T", "6758.T", "9984.T", "6861.T", "4519.T",
        # Corea (2)
        "005930.KS", "000660.KS",
        # Australia (3)
        "BHP.AX", "CSL.AX", "CBA.AX",
        # Asia HK/Singapore/India (6)
        "0700.HK", "9988.HK", "D05.SI",
        "TCS.NS", "INFY.NS", "HDFCBANK.NS",
        # Canada (3)
        "SHOP.TO", "RY.TO", "CNR.TO",
        # Brasile (3)
        "VALE3.SA", "ITUB4.SA", "WEGE3.SA",
    ]

    print("\n" + "="*60)
    print("  PORTFOLIO OPTIMIZER - MODALITÀ DEMO (dati simulati)")
    print(f"  Avviato: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"  Capitale: {CAPITAL:,.0f} €  |  Titoli: {len(DEMO_TICKERS)}")
    print("="*60 + "\n")

    # 1. Dati simulati
    print("📥 [1/6] Generazione dati sintetici...")
    price_data  = generate_synthetic_data(DEMO_TICKERS, years=10)
    ticker_info = generate_synthetic_info(DEMO_TICKERS)
    print(f"   ✓ {len(DEMO_TICKERS)} titoli, {len(price_data)} giorni di storico\n")

    # 2. Analisi tecnica
    print("📊 [2/6] Analisi tecnica e statistica...")
    tech        = TechnicalAnalyzer(price_data)
    tech_scores = tech.analyze_all(DEMO_TICKERS)
    print(f"   ✓ Completata per {len(tech_scores)} titoli\n")

    # 3. Analisi fondamentale
    print("🔍 [3/6] Analisi fondamentale...")
    fund        = FundamentalAnalyzer(ticker_info)
    fund_scores = fund.analyze_all(DEMO_TICKERS)
    print(f"   ✓ Completata per {len(fund_scores)} titoli\n")

    # 4. Pre-selezione candidati
    print("⭐ [4/6] Pre-selezione candidati...")
    pre_scorer  = StockScorer(tech_scores, fund_scores, ticker_info)
    pre_ranked  = pre_scorer.rank_stocks()
    pre_pool    = pre_ranked.head(24)["ticker"].tolist()
    print(f"   ✓ Pool: {len(pre_pool)} titoli selezionati per analisi notizie\n")

    # 5. Notizie sintetiche (demo — in produzione usa yfinance reale)
    print("📰 [5/6] Analisi notizie (simulata per demo)...")
    news_scores = generate_synthetic_news(pre_pool, ticker_info)
    pos = sum(1 for n in news_scores.values() if n["signal"] == "POSITIVO")
    neg = sum(1 for n in news_scores.values() if n["signal"] == "NEGATIVO")
    neu = len(news_scores) - pos - neg
    print(f"   ✓ Sentiment: 🟢 {pos} positivi  ⚪ {neu} neutri  🔴 {neg} negativi\n")

    # 6. Scoring finale + ottimizzazione
    print("⚖️  [6/6] Scoring con news + ottimizzazione portafoglio...")
    scorer   = StockScorer(tech_scores, fund_scores, ticker_info, news_scores)
    ranked   = scorer.rank_stocks()
    selected = scorer.select_top_stocks(ranked, min_stocks=8, max_stocks=12)
    print(f"   ✓ Selezionati {len(selected)} titoli finali\n")

    optimizer = PortfolioOptimizer(
        price_data=price_data[selected],
        scores=ranked,
        capital=CAPITAL
    )
    portfolio = optimizer.optimize()
    backtest  = optimizer.backtest(portfolio)
    print(f"   ✓ Rendimento atteso: {portfolio['expected_return']:.1%}/anno\n")

    # Output
    print("📄 Generazione report...")
    reporter = ReportGenerator(
        portfolio=portfolio,
        backtest=backtest,
        price_data=price_data,
        ticker_info=ticker_info,
        scores=ranked,
        capital=CAPITAL
    )
    reporter.print_summary()
    reporter.save_report()
    print("✅ Report HTML salvato in output/portfolio_report.html")

    print(f"\n📅 Prossima revisione: {portfolio['next_review']}")
    print("="*60)
    print("\n⚠️  NOTA: Questi sono dati SIMULATI per scopo dimostrativo.")
    print("   Per dati reali esegui: python main.py\n")


if __name__ == "__main__":
    run_demo()
