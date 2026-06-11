# RetailIQ — Data-Driven Retail Inventory Optimization
### Using Time-Series Forecasting for Demand Prediction
**Final Year Project | Dept. of Computer Science & Engineering**

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note for Prophet:** If `pip install prophet` fails, try:
> ```bash
> conda install -c conda-forge prophet
> ```
> The app works fully without Prophet — ARIMA is always available.

### 2. Place your dataset
- Put your CSV file in the same folder as `app.py`
- **Or** use the built-in `supermarket_sales.csv` (already included)
- **Or** upload any CSV via the sidebar at runtime

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Required Libraries (`requirements.txt`)

| Library | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.32 | Web UI framework |
| pandas | ≥2.0 | Data manipulation |
| numpy | ≥1.25 | Numerical computing |
| plotly | ≥5.18 | Interactive charts |
| statsmodels | ≥0.14 | ARIMA model |
| scikit-learn | ≥1.3 | MAE / RMSE metrics |
| prophet | ≥1.1.5 | Facebook Prophet forecasting |

---

## 📋 Sample Dataset Format

Your CSV must contain at minimum these columns (names are flexible — the app auto-detects):

```
Date,Product,Sales
2024-01-01,Electronics,1500.00
2024-01-01,Clothing,850.00
2024-01-02,Electronics,1320.00
...
```

### Supported column name variants
| Required Field | Accepted Names |
|----------------|----------------|
| Date | `date`, `Date`, `order_date`, `transaction_date` |
| Product | `product_line`, `product`, `Product`, `category` |
| Sales | `total`, `sales`, `Sales`, `revenue` |
| Quantity | `quantity`, `qty`, `units` |

The built-in **supermarket_sales.csv** (Kaggle Supermarket Sales dataset) works out of the box.

---

## 🗂️ App Structure

```
app.py                    ← Single-file Streamlit app (all modules included)
supermarket_sales.csv     ← Default dataset
requirements.txt          ← Python dependencies
README.md                 ← This file
```

### Module layout inside `app.py`
```
Module 1 — Data Loading & Cleaning     (load_and_clean, aggregate_by_product_date)
Module 2 — Forecasting                 (run_arima, run_prophet, metrics)
Module 3 — Inventory Optimisation      (compute_inventory)

Tab 1 — Data Overview     (schema, sample rows, missing values)
Tab 2 — EDA               (time trends, seasonality, product comparison)
Tab 3 — Forecasting       (ARIMA / Prophet, Actual vs Predicted, future chart)
Tab 4 — Inventory         (alerts, cards, reorder logic, download)
```

---

## ⚙️ Features Summary

- **Sidebar:** CSV upload, product dropdown, date range, model selector, forecast horizon, inventory params
- **Data Overview:** schema, nulls, sample table, statistical summary
- **EDA:** sales over time, 7-day MA, monthly/weekly trends, seasonal decomposition, product comparison, distribution
- **Forecasting:** ARIMA (always) + Prophet (if installed), train/test split, Actual vs Predicted, future forecast with confidence band, MAE / RMSE / MAPE, CSV download
- **Inventory:** safety stock, reorder level, suggested order qty, reorder/excess/optimal decision, risk cards, alert cards, charts, CSV download
