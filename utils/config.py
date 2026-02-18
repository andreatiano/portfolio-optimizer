"""
Configurazione centrale del Portfolio Optimizer.
Modifica questo file per personalizzare i parametri.
"""


class Config:
    # ─── UNIVERSO TITOLI — COPERTURA GLOBALE (~300 titoli) ───────

    # ── USA ── Large & Mid Cap per settore
    USA_TICKERS = [
        # Technology (25)
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL",
        "CRM", "ADBE", "AMD", "QCOM", "TXN", "INTC", "AMAT", "LRCX",
        "MU", "KLAC", "NOW", "SNOW", "PANW", "CRWD", "FTNT", "NET",
        "UBER", "LYFT",
        # Healthcare (18)
        "UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR",
        "ISRG", "BSX", "SYK", "MDT", "EW", "REGN", "VRTX", "AMGN",
        "BIIB", "GILD",
        # Finance (18)
        "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS",
        "C", "AXP", "BLK", "SCHW", "CB", "PGR", "AIG", "MET",
        "COF", "DFS",
        # Consumer Cyclical (14)
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX",
        "BKNG", "ABNB", "MAR", "HLT", "YUM", "CMG",
        # Consumer Defensive (10)
        "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "CL",
        "KMB", "GIS",
        # Industrials (12)
        "GE", "HON", "CAT", "DE", "RTX", "LMT", "NOC", "BA",
        "ITW", "EMR", "ETN", "PH",
        # Communication (7 — GOOGL e META già in Technology)
        "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR",
        # Energy (8)
        "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "PSX", "MPC",
        # Utilities (6)
        "NEE", "DUK", "SO", "D", "AEP", "EXC",
        # Real Estate (6)
        "AMT", "PLD", "EQIX", "CCI", "SPG", "O",
        # Basic Materials (5)
        "LIN", "APD", "ECL", "NEM", "FCX",
    ]

    # ── EUROPA ── Top 80 titoli per paese
    EU_TICKERS = [
        # Germania (12)
        "SAP.DE", "SIE.DE", "ALV.DE", "MRK.DE", "ADS.DE",
        "BMW.DE", "VOW3.DE", "BAS.DE", "DTE.DE", "MBG.DE",
        "RWE.DE", "DBK.DE",
        # Francia (12)
        "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AIR.PA",
        "BNP.PA", "AI.PA", "EL.PA", "RI.PA", "DG.PA",
        "VIE.PA", "ACA.PA",
        # UK (12)
        "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "GSK.L",
        "RIO.L", "BP.L", "REL.L", "LLOY.L", "BATS.L",
        "DGE.L", "CRH.L",
        # Svizzera (8)
        "NESN.SW", "NOVN.SW", "ROG.SW", "ABBN.SW", "ZURN.SW",
        "SREN.SW", "LONN.SW", "CFR.SW",
        # Olanda (6)
        "ASML.AS", "HEIA.AS", "INGA.AS", "PHIA.AS", "WKL.AS", "AD.AS",
        # Italia (6)
        "ENI.MI", "ENEL.MI", "ISP.MI", "UCG.MI", "G.MI", "STM.MI",
        # Spagna (5)
        "ITX.MC", "SAN.MC", "IBE.MC", "BBVA.MC", "REP.MC",
        # Svezia (5)
        "ATCO-A.ST", "INVE-B.ST", "VOLV-B.ST", "ERIC-B.ST", "SAND.ST",
        # Danimarca (4)
        "NOVO-B.CO", "MAERSK-B.CO", "DSV.CO", "COLO-B.CO",
        # Norvegia (3)
        "EQNR.OL", "DNB.OL", "MOWI.OL",
        # Finlandia (2)
        "NOKIA.HE", "FORTUM.HE",
        # Belgio (2)
        "UCB.BR", "KBC.BR",
        # Irlanda (2)
        "CRH.IR", "AIB.IR",
        # Lussemburgo (1)
        "STEF.PA",
    ]

    # ── GIAPPONE ── Top 25 Tokyo Stock Exchange
    JP_TICKERS = [
        "7203.T",   # Toyota
        "6758.T",   # Sony
        "9984.T",   # SoftBank
        "6861.T",   # Keyence
        "8306.T",   # Mitsubishi UFJ
        "4519.T",   # Chugai Pharma
        "9433.T",   # KDDI
        "6501.T",   # Hitachi
        "7974.T",   # Nintendo
        "4063.T",   # Shin-Etsu Chemical
        "6367.T",   # Daikin Industries
        "9432.T",   # NTT
        "8316.T",   # Sumitomo Mitsui
        "7267.T",   # Honda
        "4568.T",   # Daiichi Sankyo
        "6954.T",   # Fanuc
        "2802.T",   # Ajinomoto
        "8035.T",   # Tokyo Electron
        "4902.T",   # Konica Minolta
        "9020.T",   # JR East
        "3382.T",   # Seven & i Holdings
        "6762.T",   # TDK
        "4543.T",   # Terumo
        "8031.T",   # Mitsui & Co
        "9022.T",   # JR Tokai
    ]

    # ── COREA DEL SUD ── Top 10 KOSPI
    KR_TICKERS = [
        "005930.KS",  # Samsung Electronics
        "000660.KS",  # SK Hynix
        "051910.KS",  # LG Chem
        "005380.KS",  # Hyundai Motor
        "035420.KS",  # NAVER
        "035720.KS",  # Kakao
        "000270.KS",  # Kia Motors
        "068270.KS",  # Celltrion
        "207940.KS",  # Samsung Biologics
        "096770.KS",  # SK Innovation
    ]

    # ── AUSTRALIA ── Top 10 ASX
    AU_TICKERS = [
        "BHP.AX",   # BHP Group
        "CSL.AX",   # CSL
        "CBA.AX",   # Commonwealth Bank
        "WES.AX",   # Wesfarmers
        "NAB.AX",   # National Australia Bank
        "ANZ.AX",   # ANZ Banking
        "WBC.AX",   # Westpac
        "MQG.AX",   # Macquarie Group
        "RIO.AX",   # Rio Tinto
        "WOW.AX",   # Woolworths
    ]

    # ── HONG KONG / CINA ── Top 12
    HK_TICKERS = [
        "0700.HK",   # Tencent
        "9988.HK",   # Alibaba
        "2318.HK",   # Ping An Insurance
        "0941.HK",   # China Mobile
        "0005.HK",   # HSBC Holdings
        "1299.HK",   # AIA Group
        "3690.HK",   # Meituan
        "9618.HK",   # JD.com
        "2020.HK",   # ANTA Sports
        "0883.HK",   # CNOOC
        "1810.HK",   # Xiaomi
        "9999.HK",   # NetEase
    ]

    # ── SINGAPORE ── Top 6
    SG_TICKERS = [
        "D05.SI",   # DBS Bank
        "O39.SI",   # OCBC
        "Z74.SI",   # Singapore Telecom
        "U11.SI",   # UOB
        "C09.SI",   # City Developments
        "S58.SI",   # SATS
    ]

    # ── INDIA ── Top 12 NSE
    IN_TICKERS = [
        "RELIANCE.NS",   # Reliance Industries
        "TCS.NS",        # Tata Consultancy Services
        "INFY.NS",       # Infosys
        "HDFCBANK.NS",   # HDFC Bank
        "WIPRO.NS",      # Wipro
        "ICICIBANK.NS",  # ICICI Bank
        "HINDUNILVR.NS", # Hindustan Unilever
        "BHARTIARTL.NS", # Bharti Airtel
        "BAJFINANCE.NS", # Bajaj Finance
        "ASIANPAINT.NS", # Asian Paints
        "MARUTI.NS",     # Maruti Suzuki
        "TITAN.NS",      # Titan Company
    ]

    # ── CANADA ── Top 12 TSX
    CA_TICKERS = [
        "SHOP.TO",   # Shopify
        "RY.TO",     # Royal Bank of Canada
        "TD.TO",     # TD Bank
        "CNR.TO",    # Canadian National Railway
        "ENB.TO",    # Enbridge
        "BAM.TO",    # Brookfield Asset Management
        "CP.TO",     # Canadian Pacific
        "BNS.TO",    # Bank of Nova Scotia
        "SU.TO",     # Suncor Energy
        "ABX.TO",    # Barrick Gold
        "MFC.TO",    # Manulife Financial
        "ATD.TO",    # Alimentation Couche-Tard
    ]

    # ── BRASILE ── Top 10 B3
    BR_TICKERS = [
        "VALE3.SA",   # Vale
        "PETR4.SA",   # Petrobras
        "ITUB4.SA",   # Itaú Unibanco
        "ABEV3.SA",   # Ambev
        "WEGE3.SA",   # WEG
        "BBDC4.SA",   # Bradesco
        "RENT3.SA",   # Localiza
        "MGLU3.SA",   # Magazine Luiza
        "SUZB3.SA",   # Suzano
        "PRIO3.SA",   # PetroRio
    ]

    # ── MESSICO ── Top 5 BMV
    MX_TICKERS = [
        "AMXL.MX",   # América Móvil
        "WALMEX.MX", # Walmart Mexico
        "FMSAUBC.MX",# Fomento Económico Mexicano
        "GMEXICOB.MX",# Grupo Mexico
        "CEMEXCPO.MX",# CEMEX
    ]

    # ── SUDAFRICA ── Top 5 JSE
    ZA_TICKERS = [
        "NPN.JO",    # Naspers
        "AGL.JO",    # Anglo American
        "SOL.JO",    # Sasol
        "SBK.JO",    # Standard Bank
        "FSR.JO",    # Firstrand
    ]

    # ── Lista unificata (~300 titoli) ────────────────────────────
    DEFAULT_TICKERS = (
        USA_TICKERS + EU_TICKERS + JP_TICKERS + KR_TICKERS +
        AU_TICKERS + HK_TICKERS + SG_TICKERS + IN_TICKERS +
        CA_TICKERS + BR_TICKERS + MX_TICKERS + ZA_TICKERS
    )

    # ─── PARAMETRI ANALISI ───────────────────────────────────────
    HISTORY_YEARS = 10          # Anni di storico
    MIN_HISTORY_YEARS = 3       # Minimo per includere il titolo

    # Pesi dello scoring finale (devono sommare a 1.0)
    SCORE_WEIGHTS = {
        "fundamental": 0.45,    # Qualità fondamentale
        "stability":   0.30,    # Stabilità e bassa volatilità
        "momentum":    0.15,    # Trend di prezzo
        "diversification": 0.10 # Bonus diversificazione settoriale
    }

    # ─── PARAMETRI FONDAMENTALI ──────────────────────────────────
    # Filtri di esclusione (titoli che NON passano vengono scartati)
    MAX_PE_RATIO = 60           # P/E massimo accettabile
    MIN_ROE = 0.08              # ROE minimo (8%)
    MAX_DEBT_TO_EQUITY = 3.0   # Leva finanziaria massima
    MIN_MARKET_CAP = 5e9        # Market cap minima: 5 miliardi

    # ─── PARAMETRI VOLATILITÀ ────────────────────────────────────
    MAX_ANNUAL_VOLATILITY = 0.45    # 45% vol annua massima
    MAX_MAX_DRAWDOWN = -0.65        # -65% drawdown massimo accettabile
    VOLATILITY_PENALTY_FACTOR = 1.5 # Penalizzazione extra alta volatilità

    # ─── OTTIMIZZAZIONE PORTAFOGLIO ──────────────────────────────
    MIN_WEIGHT = 0.03           # Peso minimo per titolo (3%)
    MAX_WEIGHT = 0.15           # Peso massimo per titolo (15%)
    MAX_SECTOR_WEIGHT = 0.30    # Peso massimo per settore (30%)
    RISK_FREE_RATE = 0.035      # Tasso risk-free (3.5%)
    RISK_AVERSION = 2.0         # Coefficiente avversione al rischio (1=basso, 3=alto)

    # ─── COSTI TRANSAZIONE ───────────────────────────────────────
    BUY_COST = 1.0              # Costo acquisto in EUR
    SELL_COST = 1.0             # Costo vendita in EUR
    MIN_POSITION_SIZE = 200     # Importo minimo per posizione (per ammortizzare costi)

    # ─── REVISIONE PORTAFOGLIO ───────────────────────────────────
    # Soglie per triggare un ribilanciamento
    REBALANCE_THRESHOLD = 0.05  # Scosta 5% dal target → ribilancia
    REVIEW_INTERVAL_MONTHS = 6  # Revisione ogni 6 mesi di default

    # ─── MAPPING PAESE PER TICKER ────────────────────────────────
    TICKER_COUNTRY_MAP = {
        # Giappone
        **{t: "Japan" for t in [
            "7203.T","6758.T","9984.T","6861.T","8306.T","4519.T",
            "9433.T","6501.T","7974.T","4063.T","6367.T","9432.T",
            "8316.T","7267.T","4568.T","6954.T","2802.T","8035.T",
            "4902.T","9020.T","3382.T","6762.T","4543.T","8031.T","9022.T",
        ]},
        # Corea del Sud
        **{t: "South Korea" for t in [
            "005930.KS","000660.KS","051910.KS","005380.KS","035420.KS",
            "035720.KS","000270.KS","068270.KS","207940.KS","096770.KS",
        ]},
        # Australia
        **{t: "Australia" for t in [
            "BHP.AX","CSL.AX","CBA.AX","WES.AX","NAB.AX","ANZ.AX",
            "WBC.AX","MQG.AX","RIO.AX","WOW.AX",
        ]},
        # Hong Kong / Cina
        **{t: "China" for t in [
            "0700.HK","9988.HK","2318.HK","0941.HK","0005.HK","1299.HK",
            "3690.HK","9618.HK","2020.HK","0883.HK","1810.HK","9999.HK",
        ]},
        # Singapore
        **{t: "Singapore" for t in ["D05.SI","O39.SI","Z74.SI","U11.SI","C09.SI","S58.SI"]},
        # India
        **{t: "India" for t in [
            "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","WIPRO.NS",
            "ICICIBANK.NS","HINDUNILVR.NS","BHARTIARTL.NS","BAJFINANCE.NS",
            "ASIANPAINT.NS","MARUTI.NS","TITAN.NS",
        ]},
        # Canada
        **{t: "Canada" for t in [
            "SHOP.TO","RY.TO","TD.TO","CNR.TO","ENB.TO","BAM.TO",
            "CP.TO","BNS.TO","SU.TO","ABX.TO","MFC.TO","ATD.TO",
        ]},
        # Brasile
        **{t: "Brazil" for t in [
            "VALE3.SA","PETR4.SA","ITUB4.SA","ABEV3.SA","WEGE3.SA",
            "BBDC4.SA","RENT3.SA","MGLU3.SA","SUZB3.SA","PRIO3.SA",
        ]},
        # Messico
        **{t: "Mexico" for t in [
            "AMXL.MX","WALMEX.MX","FMSAUBC.MX","GMEXICOB.MX","CEMEXCPO.MX",
        ]},
        # Sudafrica
        **{t: "South Africa" for t in ["NPN.JO","AGL.JO","SOL.JO","SBK.JO","FSR.JO"]},
        # Europa — paesi espliciti
        **{t: "Germany"     for t in ["SAP.DE","SIE.DE","ALV.DE","MRK.DE","ADS.DE","BMW.DE","VOW3.DE","BAS.DE","DTE.DE","MBG.DE","RWE.DE","DBK.DE"]},
        **{t: "France"      for t in ["MC.PA","OR.PA","TTE.PA","SAN.PA","AIR.PA","BNP.PA","AI.PA","EL.PA","RI.PA","DG.PA","VIE.PA","ACA.PA","STEF.PA"]},
        **{t: "United Kingdom" for t in ["AZN.L","SHEL.L","HSBA.L","ULVR.L","GSK.L","RIO.L","BP.L","REL.L","LLOY.L","BATS.L","DGE.L","CRH.L"]},
        **{t: "Switzerland" for t in ["NESN.SW","NOVN.SW","ROG.SW","ABBN.SW","ZURN.SW","SREN.SW","LONN.SW","CFR.SW"]},
        **{t: "Netherlands" for t in ["ASML.AS","HEIA.AS","INGA.AS","PHIA.AS","WKL.AS","AD.AS"]},
        **{t: "Italy"       for t in ["ENI.MI","ENEL.MI","ISP.MI","UCG.MI","G.MI","STM.MI"]},
        **{t: "Spain"       for t in ["ITX.MC","SAN.MC","IBE.MC","BBVA.MC","REP.MC"]},
        **{t: "Sweden"      for t in ["ATCO-A.ST","INVE-B.ST","VOLV-B.ST","ERIC-B.ST","SAND.ST"]},
        **{t: "Denmark"     for t in ["NOVO-B.CO","MAERSK-B.CO","DSV.CO","COLO-B.CO"]},
        **{t: "Norway"      for t in ["EQNR.OL","DNB.OL","MOWI.OL"]},
        **{t: "Finland"     for t in ["NOKIA.HE","FORTUM.HE"]},
        **{t: "Belgium"     for t in ["UCB.BR","KBC.BR"]},
        **{t: "Ireland"     for t in ["CRH.IR","AIB.IR"]},
    }
    CACHE_DIR = "data/cache"
    CACHE_EXPIRY_HOURS = 24     # Cache valida per 24 ore

    # Mappatura settori → colori grafici
    SECTOR_COLORS = {
        "Technology": "#4A90E2",
        "Healthcare": "#7ED321",
        "Financial Services": "#F5A623",
        "Consumer Cyclical": "#D0021B",
        "Consumer Defensive": "#9B59B6",
        "Industrials": "#1ABC9C",
        "Energy": "#E74C3C",
        "Communication Services": "#3498DB",
        "Real Estate": "#E67E22",
        "Utilities": "#95A5A6",
        "Basic Materials": "#27AE60",
        "Unknown": "#BDC3C7",
    }

    # ═══════════════════════════════════════════════════════════
    # PARAMETRI V7 — FUNZIONALITÀ AVANZATE
    # ═══════════════════════════════════════════════════════════

    # ─── SMART REBALANCER ───────────────────────────────────────
    # Soglia minima delta peso prima di generare un ordine (es. 0.008 = 0.8%)
    REBALANCE_TOLERANCE  = 0.008
    # Importo minimo per singola operazione (EUR)
    MIN_TRADE_EUR        = 50.0
    # Aliquota sulle plusvalenze (Italia: 26%)
    TAX_RATE_GAINS       = 0.26
    # Commissioni fisse per operazione (EUR)
    BUY_COST_EUR         = 1.0
    SELL_COST_EUR        = 1.0
    # Se True, le vendite in perdita vengono eseguite per prime (loss harvesting)
    LOSS_HARVEST         = True

    # ─── FREQUENCY SCENARIO ANALYZER ────────────────────────────
    # Costo percentuale per operazione (0.001 = 0.1% = 10bps)
    TRANSACTION_COST_PCT = 0.001
    # Parametri per la modalità di revisione dinamica
    DYN_VOL_THRESHOLD    = 0.25    # Soglia volatilità rolling 20gg (annualizzata)
    DYN_DD_THRESHOLD     = -0.08   # Soglia drawdown corrente per trigger
    DYN_MIN_GAP_DAYS     = 21      # Minimo giorni tra revisioni dinamiche
    # Minimo giorni di dati per prima ottimizzazione negli scenari
    FREQ_MIN_TRAIN_DAYS  = 126     # 6 mesi

    # ─── HORIZON ESTIMATOR ──────────────────────────────────────
    # Pesi score composito per selezione orizzonte (devono sommare a 1)
    HORIZON_STABILITY_W  = 0.40   # Peso stabilità pesi
    HORIZON_SHARPE_W     = 0.40   # Peso Sharpe OOS
    HORIZON_COST_W       = 0.20   # Peso costo di revisione
    # Limiti orizzonte (giorni trading)
    HORIZON_MIN_DAYS     = 21     # Minimo: 1 mese
    HORIZON_MAX_DAYS     = 252    # Massimo: 1 anno
    # Finestre candidate (giorni trading): 1m, 2m, 3m, 6m, 9m, 12m
    HORIZON_CANDIDATES   = [21, 42, 63, 126, 189, 252]

    # ─── WALK-FORWARD VALIDATOR ──────────────────────────────────
    # Anni di dati usati per training
    WF_TRAIN_YEARS       = 3
    # Anni di test out-of-sample
    WF_TEST_YEARS        = 1
    # Avanzamento della finestra rolling (mesi)
    WF_STEP_MONTHS       = 6
    # Orizzonti da testare in run_multi_horizon()
    WF_MULTI_HORIZONS    = [1, 2, 3, 5, 10]
    # Modalità expanding window (True = TRAIN cresce; False = finestra fissa)
    WF_EXPANDING         = False

    # ─── FLAGS ATTIVAZIONE MODULI V7 ────────────────────────────
    # Puoi disabilitare moduli per velocizzare l'esecuzione
    ENABLE_FREQ_ANALYSIS    = True   # Analisi scenari frequenza
    ENABLE_WALK_FORWARD     = True   # Backtesting walk-forward
    ENABLE_SMART_REBALANCE  = True   # Revisione intelligente (solo se --portfolio-file)
    ENABLE_HORIZON_ESTIMATOR = True  # Orizzonte temporale dinamico
