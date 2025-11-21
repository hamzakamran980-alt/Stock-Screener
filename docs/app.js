const API_BASE_URL = window.API_BASE_URL || "http://localhost:8000";

const form = document.getElementById("search-form");
const tickerInput = document.getElementById("ticker-input");
const statusMessage = document.getElementById("status-message");
const periodSelect = document.getElementById("period-select");

const companyNameEl = document.getElementById("company-name");
const tickerEl = document.getElementById("ticker");
const metaEl = document.getElementById("meta");
const priceEl = document.getElementById("price");
const changeEl = document.getElementById("change");
const countryEl = document.getElementById("country");
const currencyEl = document.getElementById("currency");
const betaEl = document.getElementById("beta");
const volatilityEl = document.getElementById("volatility");
const descriptionEl = document.getElementById("description");
const ratiosTable = document.getElementById("ratios-table");
const technicalGrid = document.getElementById("technical-grid");
const predictionEl = document.getElementById("prediction");
const predictionDisclaimerEl = document.getElementById("prediction-disclaimer");
const riskSummaryEl = document.getElementById("risk-summary");

let chartInstance;
let currentTicker = "";
let latestCurrency = "USD";

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const ticker = tickerInput.value.trim().toUpperCase();
  if (!/^[A-Za-z0-9\.]{1,10}$/.test(ticker)) {
    setStatus("Please enter a valid alphanumeric ticker.", "error");
    return;
  }
  currentTicker = ticker;
  await loadAllData(ticker);
});

periodSelect.addEventListener("change", async () => {
  if (currentTicker) {
    await loadHistory(currentTicker);
  }
});

async function loadAllData(ticker) {
  setStatus(`Loading data for ${ticker}...`, "info");
  toggleForm(true);
  try {
    const [overview, ratios, prediction] = await Promise.all([
      fetchJson(`/api/stock/${ticker}/overview`),
      fetchJson(`/api/stock/${ticker}/ratios`),
      fetchJson(`/api/stock/${ticker}/prediction`),
    ]);
    updateOverview(overview);
    updateRatios(ratios);
    updateTechnicals(ratios);
    updatePrediction(prediction);
    riskSummaryEl.textContent = overview.riskSummary || "Summary unavailable.";
    await loadHistory(ticker);
    setStatus(`Showing latest data for ${ticker}.`, "success");
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Unable to load data.", "error");
  } finally {
    toggleForm(false);
  }
}

async function loadHistory(ticker) {
  const [period, interval] = periodSelect.value.split("|");
  try {
    const history = await fetchJson(
      `/api/stock/${ticker}/history?period=${period}&interval=${interval}`
    );
    renderChart(history.prices || []);
  } catch (error) {
    console.error(error);
    setStatus("Unable to load price history.", "error");
  }
}

function setStatus(message, type = "info") {
  statusMessage.textContent = message;
  statusMessage.className = type;
}

function toggleForm(disabled) {
  form.querySelector("button").disabled = disabled;
  tickerInput.disabled = disabled;
}

async function fetchJson(path) {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    const message = await response.text();
    let detail = "Request failed.";
    try {
      const parsed = JSON.parse(message);
      detail = parsed.detail || parsed.message || detail;
    } catch {
      if (message) detail = message;
    }
    throw new Error(detail);
  }
  return response.json();
}

function updateOverview(data) {
  latestCurrency = data.currency || latestCurrency;
  companyNameEl.textContent = data.companyName || "—";
  tickerEl.textContent = data.ticker || "—";
  metaEl.textContent = [data.exchange, data.sector, data.industry].filter(Boolean).join(" · ") || "—";
  priceEl.textContent = formatCurrency(data.price, data.currency);
  changeEl.textContent = formatChange(data.change, data.changePercent);
  changeEl.className = `change ${getChangeClass(data.change)}`;
  countryEl.textContent = data.country || "—";
  currencyEl.textContent = data.currency || "—";
  betaEl.textContent = formatNumber(data.beta);
  volatilityEl.textContent = data.volatility ? `${(data.volatility * 100).toFixed(2)}%` : "—";
  descriptionEl.textContent = data.description || "Business summary unavailable.";
}

function updateRatios(data) {
  const rows = [
    ["Market Cap", data.marketCap ? formatMarketCap(data.marketCap) : "—"],
    ["P/E (Trailing)", formatNumber(data.trailingPE)],
    ["P/E (Forward)", formatNumber(data.forwardPE)],
    ["PEG Ratio", formatNumber(data.pegRatio)],
    ["Price / Book", formatNumber(data.priceToBook)],
    ["Price / Sales (TTM)", formatNumber(data.priceToSalesTTM)],
    ["Dividend Yield", formatPercent(data.dividendYield)],
    ["Payout Ratio", formatPercent(data.payoutRatio)],
    ["Beta", formatNumber(data.beta)],
    ["EPS (TTM)", formatNumber(data.epsTTM)],
    ["Return on Equity", formatPercent(data.returnOnEquity)],
    ["Profit Margin", formatPercent(data.profitMargin)],
    ["Operating Margin", formatPercent(data.operatingMargin)],
    ["Gross Margin", formatPercent(data.grossMargin)],
    ["Current Ratio", formatNumber(data.currentRatio)],
    ["Quick Ratio", formatNumber(data.quickRatio)],
    ["Total Debt", data.totalDebt ? formatCurrency(data.totalDebt, latestCurrency) : "—"],
    ["Debt / Equity", formatNumber(data.debtToEquity)],
    ["52 Week High", formatNumber(data.week52High)],
    ["52 Week Low", formatNumber(data.week52Low)],
    ["50 Day MA", formatNumber(data.ma50)],
    ["200 Day MA", formatNumber(data.ma200)],
  ];

  ratiosTable.innerHTML = rows
    .map(
      ([label, value]) => `
      <tr>
        <td>${label}</td>
        <td>${value ?? "—"}</td>
      </tr>`
    )
    .join("");
}

function updateTechnicals(data) {
  const price = priceEl.textContent;
  const metrics = [
    { label: "SMA 20", key: "sma20", signal: data.sma20Signal },
    { label: "SMA 50", key: "sma50", signal: data.sma50Signal },
    { label: "SMA 200", key: "sma200", signal: data.sma200Signal },
    { label: "EMA 20", key: "ema20", signal: data.ema20Signal },
    { label: "EMA 50", key: "ema50", signal: data.ema50Signal },
    { label: "EMA 200", key: "ema200" },
    { label: "RSI (14)", key: "rsi14", signal: data.rsiSignal },
    { label: "MACD", key: "macd", signalMessage: data.macdSignalMessage },
  ];

  technicalGrid.innerHTML = metrics
    .map((metric) => {
      const value = formatNumber(data[metric.key]);
      const signalClass = metric.signal ? `signal-${metric.signal}` : "";
      const signalText = metric.signalMessage || metric.signal || "";
      return `
        <div class="tech-card">
          <span>${metric.label}</span>
          <div class="tech-value ${signalClass}">${value || "—"}</div>
          <small>${signalText}</small>
        </div>`;
    })
    .join("");
}

function updatePrediction(prediction) {
  const { projections = {}, disclaimer } = prediction;
  const rows = Object.entries(projections).map(([label, values]) => {
    const horizon = label === "1d" ? "1 Day" : "1 Month";
    return `
      <div>
        <strong>${horizon}:</strong>
        ${formatNumber(values.point)} (range ${formatNumber(values.low)} – ${formatNumber(values.high)})
      </div>`;
  });
  predictionEl.innerHTML = rows.join("") || "<p>No projection data available.</p>";
  predictionDisclaimerEl.textContent = disclaimer || "";
}

function renderChart(prices) {
  const ctx = document.getElementById("price-chart");
  const labels = prices.map((item) => new Date(item.date).toLocaleDateString());
  const dataPoints = prices.map((item) => item.close);
  if (chartInstance) {
    chartInstance.destroy();
  }
  chartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Close",
          data: dataPoints,
          borderColor: "rgba(15, 98, 254, 1)",
          backgroundColor: "rgba(15, 98, 254, 0.1)",
          tension: 0.25,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { ticks: { maxTicksLimit: 8 } },
        y: { beginAtZero: false },
      },
    },
  });
}

function formatCurrency(value, currency = "USD") {
  if (!value && value !== 0) return "—";
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency,
      maximumFractionDigits: 2,
    }).format(value);
  } catch {
    return `$${Number(value).toFixed(2)}`;
  }
}

function formatMarketCap(value) {
  if (!value) return "—";
  const units = [
    { value: 1e12, suffix: "T" },
    { value: 1e9, suffix: "B" },
    { value: 1e6, suffix: "M" },
  ];
  for (const unit of units) {
    if (value >= unit.value) {
      return `$${(value / unit.value).toFixed(2)}${unit.suffix}`;
    }
  }
  return formatCurrency(value);
}

function formatPercent(value) {
  if (!value && value !== 0) return "—";
  return `${(value * 100).toFixed(2)}%`;
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const num = Number(value);
  if (Math.abs(num) >= 1000) {
    return num.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }
  return num.toFixed(2);
}

function formatChange(change, percent) {
  if (change === null || change === undefined) return "—";
  const arrow = change > 0 ? "▲" : change < 0 ? "▼" : "";
  const pct = percent ? `${percent.toFixed(2)}%` : "—";
  return `${arrow} ${change.toFixed(2)} (${pct})`;
}

function getChangeClass(change) {
  if (change > 0) return "signal-bullish";
  if (change < 0) return "signal-bearish";
  return "";
}
