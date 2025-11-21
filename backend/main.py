"""FastAPI backend providing stock fundamentals, indicators, and projections."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Stock Screener API",
    version="1.0.0",
    description="Backend service powering the AI Stock Screener frontend.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StockDataService:
    """Helper service wrapping yfinance calls and computations."""

    HISTORY_PERIOD = "5y"
    HISTORY_INTERVAL = "1d"
    REGRESSION_WINDOW = 60

    def __init__(self, ticker: str):
        self.ticker_symbol = ticker.upper()
        try:
            self.ticker = yf.Ticker(self.ticker_symbol)
        except Exception as exc:  # pragma: no cover - yfinance raises generic errors
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _validate_history(self, period: str, interval: str) -> pd.DataFrame:
        history = self.ticker.history(period=period, interval=interval)
        if history.empty:
            raise HTTPException(
                status_code=404, detail="Historical price data unavailable for ticker."
            )
        return history

    def _default_history(self) -> pd.DataFrame:
        return self._validate_history(self.HISTORY_PERIOD, self.HISTORY_INTERVAL)

    @staticmethod
    def _clean_number(value):
        if value is None:
            return None
        try:
            if isinstance(value, (float, int, np.floating, np.integer)):
                if pd.isna(value):
                    return None
                return float(value)
            if isinstance(value, str):
                return float(value)
        except (TypeError, ValueError):
            return None
        return None

    def overview(self) -> Dict:
        info = self._safe_info()
        fast_info = getattr(self.ticker, "fast_info", {}) or {}
        price = fast_info.get("lastPrice") or info.get("currentPrice")
        previous_close = fast_info.get("previousClose") or info.get("previousClose")
        change = None
        change_percent = None
        if price is not None and previous_close:
            change = price - previous_close
            if previous_close:
                change_percent = (change / previous_close) * 100

        history = self._default_history()
        volatility = (
            history["Close"].pct_change().dropna().std() * np.sqrt(252)
            if len(history) > 1
            else None
        )

        summary = self._summarize(info, price, volatility, history)

        return {
            "ticker": self.ticker_symbol,
            "companyName": info.get("longName") or info.get("shortName"),
            "price": self._clean_number(price),
            "currency": fast_info.get("currency") or info.get("currency"),
            "exchange": info.get("fullExchangeName") or info.get("exchange"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "change": self._clean_number(change),
            "changePercent": self._clean_number(change_percent),
            "description": info.get("longBusinessSummary"),
            "beta": self._clean_number(info.get("beta")),
            "volatility": self._clean_number(volatility),
            "riskSummary": summary,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def ratios(self) -> Dict:
        info = self._safe_info()
        history = self._default_history()
        ratios = {
            "marketCap": self._clean_number(info.get("marketCap")),
            "trailingPE": self._clean_number(info.get("trailingPE")),
            "forwardPE": self._clean_number(info.get("forwardPE")),
            "pegRatio": self._clean_number(info.get("pegRatio")),
            "priceToBook": self._clean_number(info.get("priceToBook")),
            "priceToSalesTTM": self._clean_number(info.get("priceToSalesTrailing12Months")),
            "dividendYield": self._clean_number(info.get("dividendYield")),
            "payoutRatio": self._clean_number(info.get("payoutRatio")),
            "beta": self._clean_number(info.get("beta")),
            "epsTTM": self._clean_number(info.get("trailingEps")),
            "returnOnEquity": self._clean_number(info.get("returnOnEquity")),
            "profitMargin": self._clean_number(info.get("profitMargins")),
            "operatingMargin": self._clean_number(info.get("operatingMargins")),
            "grossMargin": self._clean_number(info.get("grossMargins")),
            "currentRatio": self._clean_number(info.get("currentRatio")),
            "quickRatio": self._clean_number(info.get("quickRatio")),
            "totalDebt": self._clean_number(info.get("totalDebt")),
            "debtToEquity": self._clean_number(info.get("debtToEquity")),
            "week52High": self._clean_number(info.get("fiftyTwoWeekHigh")),
            "week52Low": self._clean_number(info.get("fiftyTwoWeekLow")),
            "ma50": self._clean_number(info.get("fiftyDayAverage")),
            "ma200": self._clean_number(info.get("twoHundredDayAverage")),
        }
        indicators = self._technical_indicators(history)
        ratios.update(indicators)
        return ratios

    def history(self, period: str, interval: str) -> List[Dict[str, float]]:
        history = self._validate_history(period, interval)
        dataset = []
        for index, row in history.iterrows():
            dataset.append(
                {
                    "date": index.isoformat(),
                    "open": self._clean_number(row["Open"]),
                    "high": self._clean_number(row["High"]),
                    "low": self._clean_number(row["Low"]),
                    "close": self._clean_number(row["Close"]),
                    "volume": self._clean_number(row["Volume"]),
                }
            )
        return dataset

    def prediction(self) -> Dict:
        history = self._default_history()
        closes = history["Close"].dropna()
        if closes.empty:
            raise HTTPException(status_code=404, detail="Unable to compute predictions.")
        regression = self._regression_projection(closes)
        ema_projection = self._ema_projection(closes)
        volatility = closes.pct_change().dropna().tail(self.REGRESSION_WINDOW).std()

        horizons = {"1d": 1, "1m": 21}
        projections = {}
        for label, days in horizons.items():
            reg_price = regression(days)
            ema_price = ema_projection(days)
            point_estimate = np.mean([reg_price, ema_price])
            vol_factor = (
                np.exp(volatility * np.sqrt(days)) if volatility and days > 0 else 1
            )
            low = point_estimate / vol_factor
            high = point_estimate * vol_factor
            projections[label] = {
                "point": self._clean_number(point_estimate),
                "low": self._clean_number(low),
                "high": self._clean_number(high),
            }

        return {
            "disclaimer": "Educational projection only. Not investment advice.",
            "projections": projections,
        }

    def _safe_info(self) -> Dict:
        try:
            info = self.ticker.get_info()
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Unable to fetch ticker info.") from exc
        if not info:
            raise HTTPException(status_code=404, detail="Ticker not found.")
        return info

    def _technical_indicators(self, history: pd.DataFrame) -> Dict:
        closes = history["Close"].dropna()
        results = {}
        for window in (20, 50, 200):
            if len(closes) >= window:
                sma = closes.rolling(window).mean().iloc[-1]
                key = f"sma{window}"
                results[key] = self._clean_number(sma)

        for span in (20, 50, 200):
            if len(closes) >= span:
                ema = closes.ewm(span=span, adjust=False).mean().iloc[-1]
                key = f"ema{span}"
                results[key] = self._clean_number(ema)

        if len(closes) >= 15:
            results["rsi14"] = self._clean_number(self._rsi(closes, window=14))

        macd, signal = self._macd(closes)
        results["macd"] = self._clean_number(macd)
        results["macdSignal"] = self._clean_number(signal)

        price = closes.iloc[-1]
        sentiment = self._technical_sentiment(results, price)
        results.update(sentiment)
        return results

    @staticmethod
    def _rsi(series: pd.Series, window: int = 14) -> float:
        delta = series.diff()
        up = delta.clip(lower=0).rolling(window).mean()
        down = -delta.clip(upper=0).rolling(window).mean()
        down = down.replace(0, np.nan)
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    @staticmethod
    def _macd(series: pd.Series) -> Tuple[float, float]:
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return float(macd.iloc[-1]), float(signal.iloc[-1])

    def _technical_sentiment(self, indicators: Dict, price: float) -> Dict:
        sentiment = {}
        if price and indicators.get("sma50") and indicators.get("sma200"):
            if price > indicators["sma50"] > indicators["sma200"]:
                sentiment["trendMessage"] = "Bullish trend (price above 50 & 200 day SMA)."
            elif price < indicators["sma50"] < indicators["sma200"]:
                sentiment["trendMessage"] = "Bearish trend (price below 50 & 200 day SMA)."
            else:
                sentiment["trendMessage"] = "Mixed trend signals."

        def classify_moving_average(value):
            if value is None or not price:
                return "neutral"
            if price > value:
                return "bullish"
            if price < value:
                return "bearish"
            return "neutral"

        for key in ("sma20", "sma50", "sma200", "ema20", "ema50"):
            sentiment[f"{key}Signal"] = classify_moving_average(indicators.get(key))

        rsi = indicators.get("rsi14")
        if rsi is not None:
            if rsi > 70:
                sentiment["rsiSignal"] = "overbought"
            elif rsi < 30:
                sentiment["rsiSignal"] = "oversold"
            else:
                sentiment["rsiSignal"] = "neutral"

        macd = indicators.get("macd")
        signal = indicators.get("macdSignal")
        if macd is not None and signal is not None:
            if macd > signal:
                sentiment["macdSignalMessage"] = "MACD above signal (bullish momentum)."
            elif macd < signal:
                sentiment["macdSignalMessage"] = "MACD below signal (bearish momentum)."
            else:
                sentiment["macdSignalMessage"] = "MACD equals signal (neutral)."
        return sentiment

    def _regression_projection(self, closes: pd.Series):
        window = min(self.REGRESSION_WINDOW, len(closes))
        recent = np.log(closes.tail(window).values)
        x = np.arange(len(recent))
        slope, intercept = np.polyfit(x, recent, 1)

        def project(days_ahead: int) -> float:
            index = len(recent) - 1 + days_ahead
            log_price = intercept + slope * index
            return float(np.exp(log_price))

        return project

    def _ema_projection(self, closes: pd.Series):
        window = min(self.REGRESSION_WINDOW, len(closes))
        ema = closes.ewm(span=20, adjust=False).mean().iloc[-1]
        avg_return = closes.pct_change().tail(window).mean()

        def project(days_ahead: int) -> float:
            if avg_return is None or np.isnan(avg_return):
                avg_return_local = 0
            else:
                avg_return_local = avg_return
            projected = ema * (1 + avg_return_local) ** max(days_ahead, 0)
            return float(projected)

        return project

    def _summarize(self, info: Dict, price: float, volatility: float, history: pd.DataFrame) -> str:
        beta = info.get("beta")
        trailing_pe = info.get("trailingPE")
        ma50 = info.get("fiftyDayAverage")
        ma200 = info.get("twoHundredDayAverage")
        lines = []

        if beta:
            if beta > 1.2:
                lines.append("Higher volatility than the market.")
            elif beta < 0.8:
                lines.append("Lower volatility profile versus the market.")
            else:
                lines.append("Volatility roughly in-line with the market.")
        elif volatility:
            if volatility > 0.4:
                lines.append("Significant realized volatility in recent trading.")

        market_pe = 24  # simple proxy for comparison
        if trailing_pe:
            if trailing_pe > market_pe * 1.1:
                lines.append("P/E sits above broad market averages (richly valued).")
            elif trailing_pe < market_pe * 0.9:
                lines.append("P/E below market levels (potential value setup).")
            else:
                lines.append("Valuation (P/E) aligns with wider market.")

        if price and ma50 and ma200:
            if price > ma50 > ma200:
                lines.append("Price trades above both 50 and 200 day averages (positive trend).")
            elif price < ma50 < ma200:
                lines.append("Price below 50 and 200 day averages (negative trend).")
            else:
                lines.append("Price hovering near key moving averages.")

        if not lines:
            return "Additional data required for summary."
        return " ".join(lines)


@app.get("/api/stock/{ticker}/overview")
def get_overview(ticker: str):
    service = StockDataService(ticker)
    return service.overview()


@app.get("/api/stock/{ticker}/ratios")
def get_ratios(ticker: str):
    service = StockDataService(ticker)
    return service.ratios()


@app.get("/api/stock/{ticker}/history")
def get_history(ticker: str, period: str = "1y", interval: str = "1d"):
    service = StockDataService(ticker)
    return {"prices": service.history(period, interval)}


@app.get("/api/stock/{ticker}/prediction")
def get_prediction(ticker: str):
    service = StockDataService(ticker)
    return service.prediction()
