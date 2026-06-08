# AI Stock Research Assistant

A polished Streamlit app that turns stock research into a conversational, multi-agent workflow. Ask natural-language questions like "Should I invest in NVDA?", "Why did TSLA drop today?", or "Compare AAPL vs MSFT", and the app routes the request through data, sentiment, technical, risk, and analyst agents before producing a grounded response.

The project still includes the classic dashboard report flow, so it works both as a chat-first research assistant and as a quick ticker dashboard.

## What It Does

- Conversational chat UI for stock questions, follow-ups, catalyst questions, beginner explanations, technical checks, risk checks, and comparisons.
- LangGraph workflow orchestration with shared state, routing, conditional branches, deeper-risk paths, and a critique/refine loop.
- Multi-agent research layer with Bull Analyst, Bear Analyst, Technical Analyst, Portfolio Manager, and Critique Reviewer roles.
- Final conversational synthesis using DeepSeek when configured, with deterministic fallback output when no API key is available.
- Local FinBERT sentiment analysis over recent yfinance headlines.
- yfinance company snapshot, price history, market data, and recent news.
- Technical indicators including RSI, MACD, moving averages, annualized volatility, and rolling volatility charts.
- Risk assessment from volatility, drawdown, trend weakness, and optional high-risk deep dive notes.
- Quick Research tab for the original ticker-based report workflow.
- Dashboard tab that can render either the latest chat result or the latest Quick Research report.

## Chat Feature

The Chat tab is the main new experience. A user types a market question, and `app.py` builds an initial workflow state containing the question, timeframe, interval, optional focus ticker, session memory, and status events. It then invokes the LangGraph workflow from `graph/workflow.py`.

The workflow first calls `graph/router.py`, which extracts ticker symbols and classifies the request into intents such as:

- `investment_thesis`
- `catalyst_analysis`
- `technical_analysis`
- `risk_analysis`
- `beginner_explanation`
- `compare`
- `unknown`

For a normal single-stock question, the graph fetches stock data and news, runs FinBERT sentiment, computes technical indicators, evaluates risk, generates a deterministic investment memo, runs the analyst agents, creates a final answer, and then asks the critique agent whether the response has enough evidence. If the critique fails, the graph runs one refinement pass.

For a comparison question, such as "AAPL vs MSFT", the router sends the workflow down a comparison branch. The app fetches both company packages and renders a side-by-side dashboard instead of running the full single-stock analysis.

## How The Agents Work Together

The agents in `agents/analysts.py` are deterministic analyst functions. They do not hallucinate raw facts or fetch their own data. Instead, they read the structured evidence already produced by the workflow and turn it into concise findings.

| Agent | Role | Inputs | Output |
|-------|------|--------|--------|
| Bull Analyst | Builds the strongest positive case | sentiment, technicals, trend structure | `bull_findings` |
| Bear Analyst | Builds the strongest downside case | risk, sentiment, volatility, drawdown | `bear_findings` |
| Technical Analyst | Interprets momentum and trend | RSI, MACD, MA20, MA50, technical summary | `technical_findings` |
| Portfolio Manager | Creates the recommendation anchor | sentiment plus risk level | `recommendation`, `confidence` |
| Critique Reviewer | Checks whether evidence blocks are complete | bull, bear, technical findings | `critique`, `needs_refine` |

After those agents run, `analysis/final_response_generator.py` builds a structured context package and sends it to DeepSeek for natural-language synthesis if `DEEPSEEK_API_KEY` is set. If DeepSeek is not configured or unavailable, it returns a readable fallback summary using the same structured evidence.

Session memory is handled by `memory/session_memory.py`. It stores recent user and assistant messages, the last tickers, and the last conclusion inside Streamlit session state. This keeps follow-up questions lightweight without requiring a database.

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

Optional: copy the environment template.

```bash
copy .env.example .env
```

For macOS / Linux:

```bash
cp .env.example .env
```

The first FinBERT run downloads the model from HuggingFace. DeepSeek is optional; without it, the app still works with deterministic fallback synthesis.

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `DEEPSEEK_API_KEY` | Optional | Enables DeepSeek conversational synthesis |
| `DEEPSEEK_BASE_URL` | Optional | Defaults to `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | Optional | Defaults to `deepseek-chat` |
| `FINBERT_MODEL` | Optional | Defaults to `ProsusAI/finbert` |
| `LOCAL_CHAT_MODEL` | Optional | Defaults to `google/flan-t5-base` for optional local LLM loader |

## How To Run

```bash
streamlit run app.py
```

Open the URL shown in the terminal, usually `http://localhost:8501`.

## Example Prompts

- `Should I invest in NVDA?`
- `Why did TSLA drop today?`
- `Compare AAPL vs MSFT`
- `What are the risks for AMD?`
- `Explain the RSI for META like I am new to investing`
- `Is GOOGL bullish or bearish right now?`

## Architecture

```text
app.py                      # Streamlit chat, quick research, dashboard, workflow invocation
agents/
  analysts.py               # Bull, Bear, Technical, PM, and Critique deterministic agents
  llm.py                    # Optional local HuggingFace/LangChain LLM loader
analysis/
  final_response_generator.py # DeepSeek synthesis plus fallback response generation
  memo_generator.py         # Deterministic investment memo
  risk_analysis.py          # Volatility/drawdown/trend risk scoring
  sentiment.py              # FinBERT sentiment pipeline
  technicals.py             # RSI, MACD, moving averages, volatility
data/
  news_fetcher.py           # Recent news from yfinance
  stock_data.py             # Price history and company fundamentals from yfinance
graph/
  router.py                 # Ticker extraction and intent classification
  state.py                  # LangGraph shared state schema
  workflow.py               # LangGraph nodes and conditional edges
memory/
  session_memory.py         # Lightweight session-local chat memory
prompts/
  agent_prompts.py          # Prompt templates kept for agent design/reference
tools/
  research_tools.py         # LangChain tool wrappers around project functions
ui/
  components.py             # Reusable Streamlit UI components
  dashboard.py              # Dashboard tabs, charts, comparison view
utils/
  config.py                 # Environment variables and constants
  helpers.py                # Formatting and retry helper
  logger.py                 # Logging setup
assets/
  style.css                 # Custom Streamlit styling
```

## Data Flow

1. `app.py` receives a chat message or Quick Research ticker.
2. `graph/router.py` extracts tickers and classifies intent.
3. `graph/workflow.py` decides whether to run comparison or single-stock research.
4. `data/stock_data.py` and `data/news_fetcher.py` gather market data.
5. `analysis/sentiment.py`, `analysis/technicals.py`, and `analysis/risk_analysis.py` compute evidence.
6. `analysis/memo_generator.py` creates a deterministic memo.
7. `agents/analysts.py` creates bull, bear, technical, and portfolio-manager findings.
8. `analysis/final_response_generator.py` synthesizes the final answer with DeepSeek or fallback logic.
9. `agents/analysts.py` critique logic decides whether a refinement pass is needed.
10. `ui/dashboard.py` renders charts, metrics, memo, sentiment, technicals, and comparison output.

## Tech Stack

| Layer | Tools |
|-------|-------|
| UI | Streamlit, custom CSS |
| Workflow | LangGraph |
| Agent/tool wrappers | LangChain Core, LangChain Community |
| Data | yfinance, pandas, requests |
| AI | FinBERT via HuggingFace Transformers, optional DeepSeek |
| Charts | Plotly |
| Config | python-dotenv |

## Interview Pitch

This app is a strong interview project because it demonstrates more than a simple dashboard. It shows a full agentic research pipeline: intent routing, structured state, tool-style data gathering, deterministic evidence agents, LLM synthesis, critique/refinement, memory, and a polished Streamlit interface.

For a deeper file-by-file explanation, see `APP_FILE_GUIDE.txt`.

## Disclaimer

This project is for educational purposes only. It does not provide investment advice. Always do your own research and consult a licensed financial advisor.

## License

MIT
