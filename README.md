## AI Stock Screener

A single-page stock screener experience powered by a FastAPI backend (yfinance data) and a lightweight frontend that can be hosted directly on GitHub Pages.

### Features
- Live overview card with change indicators, fundamentals, and company summary.
- Fundamentals table highlighting valuation, profitability, leverage, and liquidity metrics.
- Technical indicator grid with SMA/EMA, RSI, and MACD signals.
- Interactive Chart.js price history with preset periods.
- AI-inspired projections combining regression, EMA trend, and volatility ranges for 1-day and 1-month horizons.
- Natural-language risk summary based on volatility, valuation, and trend context.

### Project Structure
```
Stock-Screener/
├── backend/
│   └── main.py          # FastAPI application + analytics helpers
├── docs/                # Static frontend to host via GitHub Pages
│   ├── index.html
│   ├── style.css
│   └── app.js
├── requirements.txt     # Backend Python dependencies
└── README.md
```

### Backend Setup (FastAPI + yfinance)
1. **Install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Run the API locally**
   ```bash
   uvicorn backend.main:app --reload
   ```
   The service defaults to `http://127.0.0.1:8000`.

3. **Endpoints**
   - `GET /api/stock/{ticker}/overview` – ticker metadata, current pricing, summary.
   - `GET /api/stock/{ticker}/ratios` – fundamentals, key ratios, technical indicators.
   - `GET /api/stock/{ticker}/history?period=1y&interval=1d` – OHLCV history for charts.
   - `GET /api/stock/{ticker}/prediction` – blended projection for 1 day and 1 month.

### Frontend Setup (docs/ for GitHub Pages)
1. Enable GitHub Pages for the repository with **Branch: `main`** and **Folder: `/docs`**.
2. Update the backend URL if necessary. At the top of `docs/app.js`, change:
   ```javascript
   const API_BASE_URL = "https://your-backend-hostname";
   ```
   When running locally, leave it as `http://localhost:8000`.
3. Open `docs/index.html` in a browser (or the published GitHub Pages URL) to use the app.

### Deployment Notes
- Deploy the FastAPI backend to a service such as Render, Railway, Fly.io, or Azure App Service. Run `uvicorn backend.main:app` as the start command and set any required environment variables.
- Ensure the deployed backend allows CORS for your GitHub Pages domain (already configured for `*`).
- No database is required; all data is fetched live via yfinance.

### Disclaimer
All analytics and projections are for educational purposes only and should not be considered investment advice.
