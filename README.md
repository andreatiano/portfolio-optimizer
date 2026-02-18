# Portfolio Optimizer v7

Sistema avanzato di ottimizzazione e revisione dinamica del portafoglio azionario.

## Novità v7

| Modulo | File | Descrizione |
|--------|------|-------------|
| `FrequencyScenarioAnalyzer` | `optimization/frequency_analyzer.py` | Confronta mensile/trimestrale/semestrale/annuale/dinamico |
| `SmartRebalancer` | `optimization/smart_rebalancer.py` | Revisione intelligente con stato precedente |
| `HorizonEstimator` | `optimization/horizon_estimator.py` | Stima orizzonte temporale via analisi statistica |
| `WalkForwardValidator` | `optimization/walk_forward.py` | Backtesting iterativo rolling OOS |

## Installazione

```bash
pip install -r requirements.txt
```

## Utilizzo

```bash
# Analisi completa
python main.py --capital 50000

# Senza analisi scenari frequenza (più veloce)
python main.py --no-freq-analysis

# Senza walk-forward (più veloce)
python main.py --no-walk-forward

# Walk-forward su orizzonti multipli (1,2,3,5,10 anni training)
python main.py --multi-horizon

# Revisione intelligente del portafoglio esistente
python main.py --portfolio-file portfolio_example.json --capital 15000

# Ticker specifici
python main.py --tickers AAPL MSFT NVDA --capital 20000 --period 15y
```

## Formato portfolio-file

```json
{
  "AAPL": {"quantity": 10, "avg_price": 145.50, "current_price": 195.00},
  "MSFT": {"quantity":  5, "avg_price": 310.00, "current_price": 430.00}
}
```

## Struttura directory

```
portfolio_optimizer/
├── main.py
├── portfolio_example.json
├── optimization/
│   ├── portfolio.py
│   ├── frequency_analyzer.py   ← NUOVO
│   ├── smart_rebalancer.py     ← NUOVO
│   ├── horizon_estimator.py    ← NUOVO
│   └── walk_forward.py         ← NUOVO
├── analysis/  data/  output/  utils/
```

## Disabilitare moduli in config.py

```python
ENABLE_FREQ_ANALYSIS     = False  # Salta analisi frequenza
ENABLE_WALK_FORWARD      = False  # Salta walk-forward
ENABLE_SMART_REBALANCE   = False  # Salta revisione intelligente
ENABLE_HORIZON_ESTIMATOR = False  # Usa orizzonte euristico
```
