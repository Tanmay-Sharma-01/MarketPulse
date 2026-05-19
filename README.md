# AI Stock Research Assistant

A polished, modular **Streamlit** application that generates automated equity research reports for any US ticker. Enter a symbol (e.g. `AAPL`, `NVDA`, `TSLA`) and receive company fundamentals, recent news, **FinBERT** sentiment analysis, technical indicators, risk scoring, and a deterministic investment memo — all using **free data sources** and **local AI inference**.

## Features

- **Company snapshot** — market cap, P/E, sector, 52-week range, revenue growth
- **Price charts** — candlestick/line with 20/50-day moving averages (Plotly)
- **News feed** — recent headlines via yfinance
- **AI sentiment** — ProsusAI FinBERT (runs locally via HuggingFace Transformers)
- **Technical analysis** — RSI, MACD, moving averages, volatility
- **Risk assessment** — volatility, drawdown, trend weakness → low / moderate / high
- **Investment memo** — analyst-style report from template logic (no external LLM API)

## Screenshots

<!-- Add screenshots after running locally -->
| Overview | Technicals | Sentiment | Memo |
|----------|------------|-----------|------|
| _placeholder_ | _placeholder_ | _placeholder_ | _placeholder_ |

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
cd ai_stock_assistant
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Optional: copy environment template:

```bash
copy .env.example .env   # Windows
cp .env.example .env     # macOS / Linux
```

> **Note:** The first sentiment run downloads FinBERT (~400MB). Ensure you have internet access once; inference is local afterward.

## How to Run

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

### Example usage

1. Launch the app
2. Enter `NVDA` in the sidebar
3. Select timeframe `6mo`
4. Click **Generate Research Report**
5. Explore tabs: Overview → Technicals → Sentiment → Investment Memo

## Architecture

```
app.py                 # Entry point, orchestration, Streamlit caching
├── data/
│   ├── stock_data.py  # yfinance prices + company info
│   └── news_fetcher.py
├── analysis/
│   ├── technicals.py  # RSI, MA, MACD, volatility
│   ├── sentiment.py   # FinBERT pipeline
│   ├── risk_analysis.py
│   └── memo_generator.py
├── ui/
│   ├── dashboard.py   # Layouts & Plotly charts
│   └── components.py  # Reusable widgets
└── utils/
    ├── config.py      # Constants & .env
    ├── logger.py
    └── helpers.py     # Retry, formatting
```

### Data flow

1. `app.py` receives ticker + timeframe from sidebar
2. `data/stock_data.py` fetches OHLCV + fundamentals (with retry)
3. `data/news_fetcher.py` pulls recent headlines
4. `analysis/technicals.py` enriches price history with indicators
5. `analysis/sentiment.py` scores headlines with FinBERT
6. `analysis/risk_analysis.py` classifies risk from price + technicals
7. `analysis/memo_generator.py` assembles the investment memo
8. `ui/dashboard.py` renders tabs, charts, and components

## Tech stack

| Layer | Tools |
|-------|--------|
| UI | Streamlit, custom CSS |
| Data | yfinance, pandas |
| AI | HuggingFace Transformers (FinBERT) |
| Charts | Plotly |
| Config | python-dotenv |

## Disclaimer

This project is for **educational purposes only**. It does not provide investment advice. Always do your own research and consult a licensed financial advisor.

## License

MIT
