"""
==============================================================
  RetailIQ — Data-Driven Retail Inventory Optimization
  Using Time-Series Forecasting for Demand Prediction
  Final Year Project | Dept. of Computer Science & Engineering
==============================================================
  Run:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import datetime

# ── Modelling ──────────────────────────────────────────────
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RetailIQ — Inventory Optimization",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
# GLOBAL STYLE
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #07080f !important;
    color: #b4bcd0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d0f1a !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

/* Page header */
.page-header { margin-bottom: 1.5rem; }
.page-header h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 700;
    color: #f1f5f9; letter-spacing: -0.4px; margin: 0;
}
.page-header p { font-size: 13px; color: #64748b; margin: 4px 0 0 0; }

/* KPI card */
.kpi { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
       border-radius: 12px; padding: 18px 20px; position: relative; overflow: hidden; }
.kpi-accent { position: absolute; top:0;left:0;right:0; height:3px; border-radius:12px 12px 0 0; }
.kpi-label { font-size:11px; font-weight:500; color:#64748b; text-transform:uppercase;
             letter-spacing:1px; display:block; margin-bottom:6px; }
.kpi-val { font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700; display:block; }
.kpi-sub { font-size:11px; color:#475569; margin-top:3px; }

/* Alert badges */
.badge { font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px;
         text-transform:uppercase; letter-spacing:.5px; display:inline-block; }
.badge-danger  { background:rgba(248,113,113,.12); color:#f87171; }
.badge-warning { background:rgba(251,191,36,.12);  color:#fbbf24; }
.badge-success { background:rgba(52,211,153,.12);  color:#34d399; }
.badge-info    { background:rgba(56,189,248,.12);  color:#38bdf8; }

/* Inventory card */
.inv-card { background:rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.06);
            border-radius:12px; padding:18px 20px; margin-bottom:12px; }
.inv-head { display:flex; justify-content:space-between; align-items:center;
            margin-bottom:12px; padding-bottom:10px; border-bottom:1px solid rgba(255,255,255,.05); }
.inv-prod { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:14px; color:#f1f5f9; }
.inv-row  { display:flex; justify-content:space-between; font-size:13px; color:#64748b; padding:4px 0; }
.inv-row span:last-child { color:#e2e8f0; font-weight:500; }

/* Alert card */
.alert-card { border-radius:12px; padding:14px 18px; margin-bottom:10px; }
.alert-danger  { background:rgba(248,113,113,.08); border-left:3px solid #f87171; }
.alert-warning { background:rgba(251,191,36,.08);  border-left:3px solid #fbbf24; }
.alert-success { background:rgba(52,211,153,.08);  border-left:3px solid #34d399; }
.alert-title { font-weight:600; font-size:13px; color:#f1f5f9; margin-bottom:3px; }
.alert-body  { font-size:12px; color:#94a3b8; line-height:1.5; }

/* Chart wrapper */
.chart-wrap { background:rgba(255,255,255,.02); border:1px solid rgba(255,255,255,.06);
              border-radius:12px; padding:6px 6px 2px 6px; margin-bottom:14px; }

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background:rgba(255,255,255,.025) !important; border-radius:10px !important;
    padding:3px !important; border:1px solid rgba(255,255,255,.06) !important;
    gap:2px !important; margin-bottom:20px !important;
}
[data-testid="stTabs"] [role="tab"] {
    background:transparent !important; border:none !important; border-radius:8px !important;
    color:#64748b !important; font-size:13px !important; font-weight:500 !important;
    padding:7px 18px !important; transition:all .2s !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color:#e2e8f0 !important; background:rgba(255,255,255,.05) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background:linear-gradient(135deg,#6366f1,#4f46e5) !important;
    color:#fff !important; font-weight:600 !important;
    box-shadow:0 2px 10px rgba(99,102,241,.35) !important;
}

/* Sidebar logo */
.sb-logo { font-family:'Space Grotesk',sans-serif; font-size:17px; font-weight:700;
           color:#f1f5f9; display:flex; align-items:center; gap:8px; margin-bottom:6px; }
.sb-sub  { font-size:11px; color:#475569; margin-bottom:22px; }
.sb-divider { height:1px; background:rgba(255,255,255,.06); margin:16px 0; }
.sb-section { font-size:10px; font-weight:600; color:#475569; text-transform:uppercase;
              letter-spacing:1.5px; margin-bottom:10px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# COLOUR PALETTE & CHART HELPER
# ═══════════════════════════════════════════════════════════
PAL   = ["#6366f1","#38bdf8","#34d399","#fb923c","#f472b6","#a78bfa","#fbbf24","#2dd4bf"]
BG    = "rgba(0,0,0,0)"
GRID  = "rgba(255,255,255,0.04)"
FONT  = "#94a3b8"
TITLE = "#e2e8f0"

def stylefig(fig, title="", h=360, legend=True):
    """Apply dark theme to any plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Grotesk", size=13, color=TITLE), x=0,
                   pad=dict(l=4, t=4)),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(family="Inter", color=FONT, size=11),
        height=h, margin=dict(l=10, r=10, t=44, b=10),
        showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=FONT),
                    bordercolor="rgba(255,255,255,0.06)", borderwidth=1),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(size=11)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(size=11)),
    )
    return fig

def show(fig):
    """Render chart inside a dark card."""
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

def kpi(col, label, value, color, sub=""):
    with col:
        st.markdown(f"""
        <div class="kpi">
          <div class="kpi-accent" style="background:{color}"></div>
          <span class="kpi-label">{label}</span>
          <span class="kpi-val" style="color:{color}">{value}</span>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# ── MODULE 1: DATA LOADING & CLEANING ───────────────────
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_clean(file) -> pd.DataFrame:
    """
    Load CSV, standardise column names, parse dates,
    handle missing values, and engineer basic features.
    """
    df = pd.read_csv(file)
    # Standardise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[\s\-/]", "_", regex=True)

    # ── Identify date column ──────────────────────────────
    date_col = next((c for c in df.columns if "date" in c), None)
    if date_col is None:
        raise ValueError("No date column found in the dataset.")
    df.rename(columns={date_col: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)

    # ── Identify sales/product columns ───────────────────
    # Try 'total' or 'sales' for revenue; 'product_line' or 'product' for product
    if "total" not in df.columns and "sales" in df.columns:
        df.rename(columns={"sales": "total"}, inplace=True)
    if "product_line" not in df.columns and "product" in df.columns:
        df.rename(columns={"product": "product_line"}, inplace=True)
    if "quantity" not in df.columns and "qty" in df.columns:
        df.rename(columns={"qty": "quantity"}, inplace=True)

    # Ensure core columns exist with defaults
    for col, default in [("total", 0.0), ("quantity", 1), ("product_line", "All Products")]:
        if col not in df.columns:
            df[col] = default

    # ── Handle missing values ─────────────────────────────
    df["total"]        = pd.to_numeric(df["total"], errors="coerce").fillna(0)
    df["quantity"]     = pd.to_numeric(df["quantity"], errors="coerce").fillna(1)
    df["product_line"] = df["product_line"].fillna("Unknown").str.strip()
    df.dropna(subset=["date"], inplace=True)

    # ── Feature engineering ───────────────────────────────
    df["month"]     = df["date"].dt.month
    df["month_name"]= df["date"].dt.strftime("%b")
    df["week"]      = df["date"].dt.isocalendar().week.astype(int)
    df["dayofweek"] = df["date"].dt.day_name()
    df["year"]      = df["date"].dt.year

    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def aggregate_by_product_date(df: pd.DataFrame) -> pd.DataFrame:
    """Daily sales aggregated by product."""
    return (df.groupby(["date", "product_line"])
              .agg(sales=("total", "sum"), units=("quantity", "sum"))
              .reset_index()
              .sort_values(["product_line", "date"]))


# ═══════════════════════════════════════════════════════════
# ── MODULE 2: FORECASTING ────────────────────────────────
# ═══════════════════════════════════════════════════════════

def metrics(actual, predicted):
    """Return MAE, RMSE, MAPE as floats."""
    actual    = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual.clip(min=1)))) * 100
    return mae, rmse, mape


@st.cache_data(show_spinner=False)
def run_arima(series: pd.Series, forecast_days: int):
    """
    Fit ARIMA(1,1,1) on the time series.
    Returns: forecast_df, mae, rmse, mape
    """
    series = series.asfreq("D").fillna(method="ffill").fillna(0)
    n      = len(series)
    split  = max(int(n * 0.8), n - 30)
    train, test = series.iloc[:split], series.iloc[split:]

    # Fit on train, evaluate on test
    try:
        model = ARIMA(train, order=(1, 1, 1))
        fit   = model.fit()
        test_pred = fit.forecast(steps=len(test))
        mae, rmse, mape = metrics(test.values, test_pred.values)
    except Exception:
        mae, rmse, mape = np.nan, np.nan, np.nan

    # Refit on full series for future forecast
    try:
        full_model = ARIMA(series, order=(1, 1, 1))
        full_fit   = full_model.fit()
        forecast   = full_fit.forecast(steps=forecast_days)
        last_date  = series.index[-1]
        future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")
        fc_df      = pd.DataFrame({"date": future_idx, "forecast": forecast.values.clip(min=0)})
    except Exception as e:
        fc_df = pd.DataFrame(columns=["date", "forecast"])

    return fc_df, mae, rmse, mape, series, test, test_pred if 'test_pred' in dir() else pd.Series()


@st.cache_data(show_spinner=False)
def run_prophet(series: pd.Series, forecast_days: int):
    """
    Fit Facebook Prophet on the time series.
    Returns: forecast_df, mae, rmse, mape
    """
    if not PROPHET_OK:
        return pd.DataFrame(), np.nan, np.nan, np.nan, series, pd.Series(), pd.Series()

    df_p  = series.reset_index()
    df_p.columns = ["ds", "y"]
    df_p["y"]    = df_p["y"].clip(lower=0)

    n     = len(df_p)
    split = max(int(n * 0.8), n - 30)
    train = df_p.iloc[:split]
    test  = df_p.iloc[split:]

    try:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False, changepoint_prior_scale=0.05,
                    interval_width=0.80)
        m.fit(train)
        test_fc   = m.predict(test[["ds"]])
        mae, rmse, mape = metrics(test["y"].values, test_fc["yhat"].clip(lower=0).values)
    except Exception:
        mae, rmse, mape = np.nan, np.nan, np.nan
        test_fc = pd.DataFrame({"yhat": [0]*len(test)})

    # Full forecast
    m2 = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                 daily_seasonality=False, interval_width=0.80)
    m2.fit(df_p)
    future = m2.make_future_dataframe(periods=forecast_days)
    fc     = m2.predict(future)
    fc_future = fc[fc["ds"] > df_p["ds"].max()][["ds","yhat","yhat_lower","yhat_upper"]]
    fc_future = fc_future.rename(columns={"ds":"date","yhat":"forecast",
                                          "yhat_lower":"lower","yhat_upper":"upper"})
    fc_future["forecast"] = fc_future["forecast"].clip(lower=0)
    fc_future["lower"]    = fc_future["lower"].clip(lower=0)

    return fc_future, mae, rmse, mape, df_p, test["y"], test_fc["yhat"]


# ═══════════════════════════════════════════════════════════
# ── MODULE 3: INVENTORY OPTIMISATION ────────────────────
# ═══════════════════════════════════════════════════════════

def compute_inventory(df: pd.DataFrame, fc_map: dict,
                      lead_time: int = 7,
                      service_level: float = 0.95) -> pd.DataFrame:
    """
    For each product compute:
      - Average & std daily demand
      - Safety Stock   = Z * σ * √(lead_time)
      - Reorder Level  = avg_demand * lead_time + safety_stock
      - Suggested Reorder Qty = (avg_demand * 30) + safety_stock
      - Predicted 30-day demand from forecast
      - Decision: Reorder Required / Excess Stock / Optimal
      - Risk level
    """
    Z_MAP = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
    Z     = Z_MAP.get(service_level, 1.65)

    rows = []
    for prod in sorted(df["product_line"].unique()):
        sub      = df[df["product_line"] == prod]
        daily    = sub.groupby("date")["quantity"].sum()
        avg_d    = daily.mean()
        std_d    = daily.std(ddof=1) if len(daily) > 1 else avg_d * 0.2
        ss       = Z * std_d * np.sqrt(lead_time)
        rop      = avg_d * lead_time + ss
        reorder_qty = avg_d * 30 + ss                  # 30-day replenishment cycle

        # Forecast-based predicted demand (next 30 days)
        pred_demand = fc_map.get(prod, avg_d * 30)

        # Current "stock proxy" — last 7-day average * lead_time (no real inventory data)
        current_stock = daily.tail(7).mean() * lead_time

        if pred_demand > current_stock * 1.1:
            decision = "⚠️ Reorder Required"
            risk     = "High"
        elif pred_demand < current_stock * 0.85:
            decision = "📦 Excess Stock"
            risk     = "Low"
        else:
            decision = "✅ Optimal"
            risk     = "Medium"

        rows.append({
            "Product":            prod,
            "Avg Daily Demand":   round(avg_d, 1),
            "Std Dev":            round(std_d, 1),
            "Safety Stock":       round(ss, 0),
            "Reorder Level":      round(rop, 0),
            "Reorder Qty":        round(reorder_qty, 0),
            "30-Day Forecast":    round(pred_demand, 0),
            "Stock Proxy":        round(current_stock, 0),
            "Decision":           decision,
            "Risk":               risk,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# ── SIDEBAR ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">📦 RetailIQ</div>
    <div class="sb-sub">Inventory Optimization System · FYP</div>
    """, unsafe_allow_html=True)

    # ── Dataset upload ────────────────────────────────────
    st.markdown('<div class="sb-section">📁 Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                help="Must contain Date, Product, and Sales columns")
    if uploaded is None:
        try:
            df_raw = load_and_clean("supermarket_sales.csv")
            st.caption("Using built-in supermarket_sales.csv")
        except Exception as e:
            st.error(f"Cannot load default dataset: {e}")
            st.stop()
    else:
        try:
            df_raw = load_and_clean(uploaded)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Product selection ─────────────────────────────────
    st.markdown('<div class="sb-section">🛍️ Product Filter</div>', unsafe_allow_html=True)
    all_products = sorted(df_raw["product_line"].unique())
    selected_product = st.selectbox("Select Product", ["All Products"] + all_products)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Date range ────────────────────────────────────────
    st.markdown('<div class="sb-section">📅 Date Range</div>', unsafe_allow_html=True)
    min_date = df_raw["date"].min().date()
    max_date = df_raw["date"].max().date()
    date_range = st.date_input("Select Range",
                               value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Model selection ───────────────────────────────────
    st.markdown('<div class="sb-section">🔮 Forecasting Model</div>', unsafe_allow_html=True)
    model_options = ["ARIMA"]
    if PROPHET_OK:
        model_options.append("Prophet")
    model_choice = st.selectbox("Select Model", model_options)

    forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30, step=1)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Inventory params ──────────────────────────────────
    st.markdown('<div class="sb-section">⚙️ Inventory Params</div>', unsafe_allow_html=True)
    lead_time = st.slider("Lead Time (days)", 1, 30, 7)
    svc_level = st.select_slider("Service Level", options=[0.90, 0.95, 0.99], value=0.95)


# ═══════════════════════════════════════════════════════════
# ── APPLY FILTERS ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════
df = df_raw.copy()

if len(date_range) == 2:
    df = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])]

df_product = df.copy()  # full filtered df (all products)
if selected_product != "All Products":
    df_product = df[df["product_line"] == selected_product]

agg = aggregate_by_product_date(df)

# Series for the selected product (or overall)
if selected_product == "All Products":
    ts = df.groupby("date")["total"].sum().asfreq("D").fillna(0)
else:
    ts = (df[df["product_line"] == selected_product]
          .groupby("date")["total"].sum()
          .asfreq("D").fillna(0))

# ═══════════════════════════════════════════════════════════
# ── TOP HEADER ────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div style='margin-bottom:1.5rem;'>
  <h1 style='font-family:Space Grotesk,sans-serif;font-size:26px;font-weight:700;
             color:#f1f5f9;letter-spacing:-.4px;margin:0;'>
    📦 Retail Inventory Optimization
  </h1>
  <p style='font-size:13px;color:#64748b;margin:4px 0 0;'>
    Data-Driven Demand Forecasting &nbsp;·&nbsp;
    Time-Series Analysis &nbsp;·&nbsp; Final Year Project
  </p>
</div>
""", unsafe_allow_html=True)

# ── Global KPIs ───────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
kpi(c1, "Total Revenue",    f"₹{df['total'].sum():,.0f}",    "#6366f1", "Filtered period")
kpi(c2, "Units Sold",       f"{df['quantity'].sum():,}",      "#38bdf8", "Total qty")
kpi(c3, "Transactions",     f"{len(df):,}",                   "#34d399", "Invoice count")
kpi(c4, "Products",         f"{df['product_line'].nunique()}", "#fb923c", "Unique lines")
kpi(c5, "Date Span",        f"{(df['date'].max()-df['date'].min()).days}d",
    "#fbbf24", f"{df['date'].min().strftime('%d %b')} – {df['date'].max().strftime('%d %b %Y')}")

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# ── MAIN TABS ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════
tabs = st.tabs([
    "📋  Data Overview",
    "📊  EDA",
    "🔮  Forecasting",
    "🏭  Inventory Optimization"
])


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 1 — DATA OVERVIEW                                  ║
# ╚══════════════════════════════════════════════════════════╝
with tabs[0]:
    st.markdown("""
    <div class="page-header">
      <h2>Data Overview</h2>
      <p>Dataset summary, column types, missing values, and sample records.</p>
    </div>""", unsafe_allow_html=True)

    # ── Dataset info ──────────────────────────────────────
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("**📄 Sample Records (first 50)**")
        st.dataframe(df.head(50), use_container_width=True, height=320)

    with col_b:
        st.markdown("**📐 Dataset Info**")
        info = pd.DataFrame({
            "Column":    df.columns,
            "Dtype":     [str(d) for d in df.dtypes],
            "Non-Null":  df.notna().sum().values,
            "Nulls":     df.isna().sum().values,
            "Unique":    [df[c].nunique() for c in df.columns],
        })
        st.dataframe(info, use_container_width=True, height=320)

    st.markdown("---")

    # ── Aggregated daily sales table ──────────────────────
    with st.expander("📅 Daily Aggregated Sales (by Product)"):
        st.dataframe(agg.head(200), use_container_width=True, height=280)

    # ── Numeric summary ───────────────────────────────────
    with st.expander("📊 Statistical Summary"):
        st.dataframe(df.describe().T.round(2), use_container_width=True)

    # ── Missing value heatmap ─────────────────────────────
    with st.expander("🔍 Missing Values by Column"):
        miss = df.isna().sum().reset_index()
        miss.columns = ["Column", "Missing"]
        miss = miss[miss["Missing"] > 0]
        if miss.empty:
            st.success("✅ No missing values in the dataset.")
        else:
            fig = px.bar(miss, x="Column", y="Missing",
                         color="Missing", color_continuous_scale="Reds")
            stylefig(fig, "Missing Values per Column", h=280)
            show(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 2 — EDA                                            ║
# ╚══════════════════════════════════════════════════════════╝
with tabs[1]:
    st.markdown("""
    <div class="page-header">
      <h2>Exploratory Data Analysis</h2>
      <p>Sales trends, seasonality, product comparisons, and distribution analysis.</p>
    </div>""", unsafe_allow_html=True)

    # ── 2.1 Sales over time ───────────────────────────────
    st.markdown("#### 📈 Sales Over Time")
    daily_total = df.groupby("date")["total"].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_total["date"], y=daily_total["total"],
                             mode="lines", name="Daily Revenue",
                             line=dict(color="#6366f1", width=2),
                             fill="tozeroy", fillcolor="rgba(99,102,241,.08)"))
    # 7-day rolling avg
    roll = daily_total["total"].rolling(7).mean()
    fig.add_trace(go.Scatter(x=daily_total["date"], y=roll, mode="lines",
                             name="7-Day MA", line=dict(color="#38bdf8", width=1.5, dash="dot")))
    stylefig(fig, "Daily Revenue with 7-Day Moving Average", h=340)
    show(fig)

    # ── 2.2 Monthly trend ─────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        monthly = df.groupby(["year","month","month_name"])["total"].sum().reset_index()
        monthly = monthly.sort_values(["year","month"])
        monthly["period"] = monthly["month_name"] + " " + monthly["year"].astype(str)
        fig = px.bar(monthly, x="period", y="total",
                     color="total", color_continuous_scale="Purples",
                     text_auto=".2s")
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Revenue ₹",
                          xaxis_tickangle=-30)
        stylefig(fig, "Monthly Revenue Trend", h=320)
        show(fig)

    with c2:
        dow = df.groupby("dayofweek")["total"].sum().reset_index()
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow["dayofweek"] = pd.Categorical(dow["dayofweek"], categories=order, ordered=True)
        dow = dow.sort_values("dayofweek")
        fig = px.bar(dow, x="dayofweek", y="total",
                     color="total", color_continuous_scale="Blues", text_auto=".2s")
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Revenue ₹")
        stylefig(fig, "Revenue by Day of Week", h=320)
        show(fig)

    # ── 2.3 Product-wise demand comparison ───────────────
    st.markdown("#### 🛍️ Product-Wise Demand Comparison")
    c1, c2 = st.columns(2)
    with c1:
        prod_rev = df.groupby("product_line")["total"].sum().sort_values(ascending=True).reset_index()
        fig = px.bar(prod_rev, x="total", y="product_line", orientation="h",
                     color="product_line", color_discrete_sequence=PAL, text_auto=".2s")
        fig.update_traces(textposition="outside", marker_line_width=0, showlegend=False)
        fig.update_layout(yaxis_title="", xaxis_title="Revenue ₹")
        stylefig(fig, "Revenue by Product Line", h=320, legend=False)
        show(fig)

    with c2:
        prod_trend = df.groupby(["month","month_name","product_line"])["total"].sum().reset_index()
        prod_trend = prod_trend.sort_values("month")
        fig = px.line(prod_trend, x="month_name", y="total", color="product_line",
                      color_discrete_sequence=PAL, markers=True)
        fig.update_traces(line_width=2, marker_size=5)
        fig.update_layout(xaxis_title="", yaxis_title="Revenue ₹")
        stylefig(fig, "Monthly Revenue by Product Line", h=320)
        show(fig)

    # ── 2.4 Seasonal decomposition ───────────────────────
    st.markdown("#### 🌊 Seasonal Decomposition")
    try:
        decomp_series = df.groupby("date")["total"].sum().asfreq("D").fillna(method="ffill")
        if len(decomp_series) >= 14:
            result = seasonal_decompose(decomp_series, model="additive", period=7, extrapolate_trend="freq")
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                subplot_titles=["Observed","Trend","Seasonal","Residual"],
                                vertical_spacing=0.06)
            components = [result.observed, result.trend, result.seasonal, result.resid]
            colors     = ["#6366f1","#38bdf8","#34d399","#fb923c"]
            for i, (comp, clr) in enumerate(zip(components, colors), 1):
                fig.add_trace(go.Scatter(x=comp.index, y=comp.values, mode="lines",
                                         line=dict(color=clr, width=1.5), showlegend=False), row=i, col=1)
            fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG,
                              font=dict(family="Inter", color=FONT, size=11),
                              height=520, margin=dict(l=10,r=10,t=44,b=10),
                              title=dict(text="Seasonal Decomposition (Period = 7 days)",
                                         font=dict(family="Space Grotesk", size=13, color=TITLE), x=0))
            for i in range(1, 5):
                fig.update_xaxes(gridcolor=GRID, row=i, col=1)
                fig.update_yaxes(gridcolor=GRID, row=i, col=1)
            show(fig)
        else:
            st.info("Not enough data points for seasonal decomposition (need ≥ 14 days).")
    except Exception as e:
        st.warning(f"Decomposition skipped: {e}")

    # ── 2.5 Distribution ──────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="total", nbins=40, color_discrete_sequence=["#6366f1"])
        fig.update_traces(marker_line_width=0)
        fig.update_layout(bargap=0.05, xaxis_title="Transaction Value ₹", yaxis_title="Count",
                          showlegend=False)
        stylefig(fig, "Transaction Value Distribution", h=280)
        show(fig)

    with c2:
        fig = px.box(df, x="product_line", y="total", color="product_line",
                     color_discrete_sequence=PAL)
        fig.update_traces(showlegend=False)
        fig.update_layout(xaxis_tickangle=-25, xaxis_title="", yaxis_title="Revenue ₹")
        stylefig(fig, "Revenue Distribution per Product Line", h=280, legend=False)
        show(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 3 — FORECASTING                                    ║
# ╚══════════════════════════════════════════════════════════╝
with tabs[2]:
    st.markdown(f"""
    <div class="page-header">
      <h2>Time-Series Demand Forecasting</h2>
      <p>Model: <b>{model_choice}</b> &nbsp;·&nbsp;
         Product: <b>{selected_product}</b> &nbsp;·&nbsp;
         Horizon: <b>{forecast_days} days</b></p>
    </div>""", unsafe_allow_html=True)

    # ── Run selected model ────────────────────────────────
    with st.spinner(f"Training {model_choice} model…"):
        if model_choice == "ARIMA":
            fc_df, mae, rmse, mape, hist_series, test_actual, test_pred = run_arima(ts, forecast_days)
        else:
            fc_df, mae, rmse, mape, hist_series, test_actual, test_pred = run_prophet(ts, forecast_days)

    # ── Metrics row ───────────────────────────────────────
    accuracy = max(0.0, 100.0 - (mape if not np.isnan(mape) else 0))

    a1,a2,a3,a4 = st.columns(4)
    kpi(a1, "MAE",      f"₹{mae:,.1f}"   if not np.isnan(mae)  else "N/A", "#38bdf8", "Mean Absolute Error")
    kpi(a2, "RMSE",     f"₹{rmse:,.1f}"  if not np.isnan(rmse) else "N/A", "#a78bfa", "Root Mean Sq Error")
    kpi(a3, "MAPE",     f"{mape:.1f}%"   if not np.isnan(mape) else "N/A", "#fb923c", "Mean Abs % Error")
    kpi(a4, "Accuracy", f"{accuracy:.1f}%",                                 "#34d399", "100 – MAPE")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Actual vs Predicted (test window) ─────────────────
    if len(test_actual) > 0 and len(test_pred) > 0:
        st.markdown("#### 🔍 Actual vs Predicted (Validation Window)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(test_actual))), y=test_actual.values if hasattr(test_actual, 'values') else test_actual,
            mode="lines+markers", name="Actual",
            line=dict(color="#38bdf8", width=2), marker=dict(size=4)))
        fig.add_trace(go.Scatter(
            x=list(range(len(test_pred))), y=test_pred.values if hasattr(test_pred, 'values') else test_pred,
            mode="lines+markers", name="Predicted",
            line=dict(color="#f87171", width=2, dash="dash"), marker=dict(size=4)))
        stylefig(fig, "Actual vs Predicted — Validation Set", h=300)
        show(fig)

    # ── Future forecast chart ─────────────────────────────
    st.markdown(f"#### 📅 Future Forecast — Next {forecast_days} Days")
    if not fc_df.empty:
        fig = go.Figure()

        # Historical actual
        if model_choice == "ARIMA":
            hist_df = hist_series.reset_index()
            hist_df.columns = ["date","sales"]
        else:
            hist_df = hist_series.rename(columns={"ds":"date","y":"sales"}) if isinstance(hist_series, pd.DataFrame) else hist_series.reset_index().rename(columns={0:"sales","date":"date"})

        fig.add_trace(go.Scatter(
            x=hist_df["date"], y=hist_df["sales"],
            mode="lines", name="Historical Sales",
            line=dict(color="#38bdf8", width=1.8),
            fill="tozeroy", fillcolor="rgba(56,189,248,.05)"))

        # Confidence band (Prophet only)
        if "lower" in fc_df.columns and "upper" in fc_df.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
                y=pd.concat([fc_df["upper"], fc_df["lower"][::-1]]),
                fill="toself", fillcolor="rgba(99,102,241,.12)",
                line=dict(color="rgba(0,0,0,0)"), name="80% CI"))

        fig.add_trace(go.Scatter(
            x=fc_df["date"], y=fc_df["forecast"],
            mode="lines+markers", name=f"{model_choice} Forecast",
            line=dict(color="#6366f1", width=2.5, dash="dash"),
            marker=dict(size=4)))

        # Vertical separator
        last_hist = hist_df["date"].max()
        fig.add_vline(x=last_hist, line_dash="dot",
                      line_color="rgba(255,255,255,0.2)", line_width=1)

        stylefig(fig, f"{model_choice} Demand Forecast — {selected_product}", h=420)
        show(fig)

        # ── Download forecast CSV ─────────────────────────
        csv_buf = BytesIO()
        fc_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="⬇️  Download Forecast CSV",
            data=csv_buf.getvalue(),
            file_name=f"forecast_{selected_product.replace(' ','_')}_{model_choice}.csv",
            mime="text/csv"
        )
    else:
        st.warning("Forecast could not be generated. Try a different product or model.")

    # ── Per-product summary table ─────────────────────────
    st.markdown("#### 📦 Per-Product Demand Summary")
    prod_sum = []
    for p in sorted(df["product_line"].unique()):
        sub  = df[df["product_line"] == p].groupby("date")["total"].sum()
        avg  = sub.mean()
        pred = avg * forecast_days  # simple proxy when per-product model not run
        prod_sum.append({"Product": p,
                         "Avg Daily Revenue": f"₹{avg:,.0f}",
                         f"Est. {forecast_days}-Day Demand": f"₹{pred:,.0f}",
                         "Data Points": len(sub)})
    st.dataframe(pd.DataFrame(prod_sum), use_container_width=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 4 — INVENTORY OPTIMIZATION                         ║
# ╚══════════════════════════════════════════════════════════╝
with tabs[3]:
    st.markdown("""
    <div class="page-header">
      <h2>Inventory Optimization</h2>
      <p>Safety stock, reorder levels, EOQ, stockout risk, and demand-driven reorder decisions.</p>
    </div>""", unsafe_allow_html=True)

    # Build 30-day forecast map for each product
    fc_map = {}
    with st.spinner("Computing per-product forecasts…"):
        for p in df["product_line"].unique():
            sub_ts = (df[df["product_line"] == p]
                      .groupby("date")["quantity"].sum()
                      .asfreq("D").fillna(0))
            if len(sub_ts) < 10:
                fc_map[p] = sub_ts.mean() * 30
                continue
            try:
                if model_choice == "ARIMA":
                    fc_p, *_ = run_arima(sub_ts, 30)
                    fc_map[p] = fc_p["forecast"].sum() if not fc_p.empty else sub_ts.mean() * 30
                else:
                    fc_p, *_ = run_prophet(sub_ts, 30)
                    fc_map[p] = fc_p["forecast"].sum() if not fc_p.empty else sub_ts.mean() * 30
            except Exception:
                fc_map[p] = sub_ts.mean() * 30

    inv_df = compute_inventory(df, fc_map, lead_time=lead_time, service_level=svc_level)

    # ── Summary KPIs ──────────────────────────────────────
    n_high   = (inv_df["Risk"] == "High").sum()
    n_medium = (inv_df["Risk"] == "Medium").sum()
    n_low    = (inv_df["Risk"] == "Low").sum()
    n_reorder= inv_df["Decision"].str.contains("Reorder").sum()

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1, "High Risk Products",  str(n_high),   "#f87171", "Immediate action needed")
    kpi(c2, "Medium Risk",         str(n_medium),  "#fbbf24", "Monitor closely")
    kpi(c3, "Optimal Stock",       str(n_low),     "#34d399", "No action needed")
    kpi(c4, "Reorder Alerts",      str(n_reorder), "#6366f1", "Place orders now")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Alert cards ───────────────────────────────────────
    st.markdown("#### 🚨 Inventory Alerts")
    for _, row in inv_df.iterrows():
        if "Reorder Required" in row["Decision"]:
            cls   = "alert-danger"
            icon  = "🔴"
            title = f"{icon} {row['Product']} — Reorder Required"
            body  = (f"Predicted 30-day demand: <b>{row['30-Day Forecast']:,.0f} units</b> "
                     f"vs stock proxy: <b>{row['Stock Proxy']:,.0f} units</b>. "
                     f"Suggested reorder quantity: <b>{row['Reorder Qty']:,.0f} units</b>. "
                     f"Reorder level threshold: <b>{row['Reorder Level']:,.0f} units</b>.")
        elif "Excess Stock" in row["Decision"]:
            cls   = "alert-warning"
            icon  = "🟡"
            title = f"{icon} {row['Product']} — Excess Stock"
            body  = (f"Current stock proxy (<b>{row['Stock Proxy']:,.0f} units</b>) "
                     f"exceeds forecast demand (<b>{row['30-Day Forecast']:,.0f} units</b>). "
                     f"Consider running a promotion or delaying next order.")
        else:
            cls   = "alert-success"
            icon  = "🟢"
            title = f"{icon} {row['Product']} — Optimal"
            body  = (f"Stock levels aligned with forecasted demand. "
                     f"Safety stock: <b>{row['Safety Stock']:,.0f} units</b>. "
                     f"Next review in {lead_time} days.")
        st.markdown(f"""
        <div class="alert-card {cls}">
          <div class="alert-title">{title}</div>
          <div class="alert-body">{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Inventory cards ───────────────────────────────────
    st.markdown("#### 📦 Product Inventory Cards")
    risk_badge = {"High": "badge-danger", "Medium": "badge-warning", "Low": "badge-success"}
    col_a, col_b = st.columns(2)
    for i, (_, row) in enumerate(inv_df.iterrows()):
        with (col_a if i % 2 == 0 else col_b):
            rc = risk_badge.get(row["Risk"], "badge-info")
            st.markdown(f"""
            <div class="inv-card">
              <div class="inv-head">
                <div class="inv-prod">🛍️ {row['Product']}</div>
                <span class="badge {rc}">{row['Risk']} Risk</span>
              </div>
              <div class="inv-row">
                <span>📊 Avg Daily Demand</span><span>{row['Avg Daily Demand']} units/day</span>
              </div>
              <div class="inv-row">
                <span>🛡️ Safety Stock</span><span>{int(row['Safety Stock'])} units</span>
              </div>
              <div class="inv-row">
                <span>📍 Reorder Level</span><span>{int(row['Reorder Level'])} units</span>
              </div>
              <div class="inv-row">
                <span>📦 Suggested Reorder Qty</span><span>{int(row['Reorder Qty'])} units</span>
              </div>
              <div class="inv-row">
                <span>🔮 30-Day Forecast</span><span>{int(row['30-Day Forecast'])} units</span>
              </div>
              <div class="inv-row">
                <span>🏷️ Decision</span><span>{row['Decision']}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Inventory Metrics Visualization")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Safety Stock",  x=inv_df["Product"], y=inv_df["Safety Stock"],
                             marker_color="#38bdf8", marker_line_width=0))
        fig.add_trace(go.Bar(name="Reorder Level", x=inv_df["Product"], y=inv_df["Reorder Level"],
                             marker_color="#6366f1", marker_line_width=0))
        fig.add_trace(go.Bar(name="Reorder Qty",   x=inv_df["Product"], y=inv_df["Reorder Qty"],
                             marker_color="#34d399", marker_line_width=0))
        fig.update_layout(barmode="group", xaxis_tickangle=-25, xaxis_title="", yaxis_title="Units")
        stylefig(fig, "Safety Stock vs Reorder Level vs Suggested Order Qty", h=360)
        show(fig)

    with c2:
        risk_counts = inv_df["Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk","Count"]
        fig = px.pie(risk_counts, names="Risk", values="Count", hole=0.52,
                     color="Risk",
                     color_discrete_map={"High":"#f87171","Medium":"#fbbf24","Low":"#34d399"})
        fig.update_traces(textinfo="percent+label",
                          marker=dict(line=dict(color="#07080f", width=2)))
        stylefig(fig, "Stockout Risk Distribution", h=360)
        show(fig)

    # 30-day forecast vs stock proxy comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(name="30-Day Forecast", x=inv_df["Product"],
                         y=inv_df["30-Day Forecast"], marker_color="#6366f1", marker_line_width=0))
    fig.add_trace(go.Bar(name="Stock Proxy",     x=inv_df["Product"],
                         y=inv_df["Stock Proxy"],     marker_color="#fb923c", marker_line_width=0))
    fig.update_layout(barmode="group", xaxis_tickangle=-25, xaxis_title="", yaxis_title="Units")
    stylefig(fig, "30-Day Forecast Demand vs Estimated Current Stock", h=340)
    show(fig)

    # ── Full table & download ──────────────────────────────
    st.markdown("#### 📋 Full Inventory Table")
    st.dataframe(inv_df, use_container_width=True)

    csv_buf = BytesIO()
    inv_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️  Download Inventory Report CSV",
        data=csv_buf.getvalue(),
        file_name="inventory_optimization_report.csv",
        mime="text/csv"
    )

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:28px 0 20px;font-size:12px;color:#334155;
            border-top:1px solid rgba(255,255,255,.04);margin-top:40px;'>
  RetailIQ &nbsp;·&nbsp; Data-Driven Retail Inventory Optimization &nbsp;·&nbsp;
  Time-Series Forecasting (ARIMA / Prophet) &nbsp;·&nbsp; Final Year Project
</div>
""", unsafe_allow_html=True)
