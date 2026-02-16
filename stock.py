import streamlit as st
import pandas as pd
import yfinance as yf
import io
import requests
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

DB_NAME = "stocker"
DB_USER = "postgres"
DB_PASSWORD = "root"
DB_HOST = "localhost"

def connect_to_db():
    try:
        return psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST
        )
    except Exception as e:
        st.error(f"DB connection error: {e}")
        return None

def delete_old_model_results(symbol):
    conn = connect_to_db()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM model_results WHERE stock_symbol = %s;", (symbol,))
        conn.commit()
    except Exception as e:
        st.error(f"Failed to delete old model results: {e}")
    finally:
        conn.close()

def safe_float(x):
    try:
        if x is None:
            return None
        xf = float(x)
        return xf if not (np.isnan(xf) or np.isinf(xf)) else None
    except:
        return None

def save_full_model_result(symbol, stock_index, model_name, metrics):
    conn = connect_to_db()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            query = """
            INSERT INTO model_results (
                stock_symbol, stock_index, model_name,
                rmse_1_week, rmse_1_month, rmse_1_year,
                mape_1_week, mape_1_month, mape_1_year
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """
            horizon_keys = ['1_week', '1_month', '1_year']
            rmse_vals = [safe_float(metrics[h][0]) for h in horizon_keys]
            mape_vals = [safe_float(metrics[h][1]) for h in horizon_keys]
            params = [symbol, stock_index, model_name] + rmse_vals + mape_vals
            cur.execute(query, params)
        conn.commit()
    except Exception as e:
        st.error(f"Error saving model results: {e}")
    finally:
        conn.close()

def compute_regression_metrics(y_true, y_pred):
    if len(y_true) < 2 or len(y_pred) < 2:
        return np.nan, np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mape

def add_all_features(df):
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['month'] = df['Date'].dt.month
    for i in range(1, 6):
        df[f'lag_{i}'] = df['Close'].shift(i)
    df['rolling_mean_7'] = df['Close'].shift(1).rolling(7).mean()
    df['rolling_std_7'] = df['Close'].shift(1).rolling(7).std()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_upper'] = sma20 + (2 * std20)
    df['BB_lower'] = sma20 - (2 * std20)
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    return df

def predict_next_days(model, last_df, n_days, feature_cols):
    df_temp = last_df.copy()
    preds, dates = [], []
    for _ in range(n_days):
        last_features = df_temp[feature_cols].iloc[-1].values.reshape(1, -1)
        pred = model.predict(last_features)[0]
        last_date = df_temp['Date'].iloc[-1]
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        preds.append(float(pred))
        dates.append(next_date)
        new_row = pd.DataFrame({'Date': [next_date], 'Close': [pred]})
        df_temp = pd.concat([df_temp, new_row], ignore_index=True)
        df_temp = add_all_features(df_temp)
        df_temp.bfill(inplace=True)
    return preds, dates

@st.cache_data(show_spinner=False)
def load_all_nse_symbols():
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbols = df['SYMBOL'].str.strip().tolist()
        return [s + ".NS" for s in symbols]
    except Exception as e:
        st.error(f"Failed to load NSE symbols: {e}")
        return []

@st.cache_data(show_spinner=False)
def fetch_stock_data(symbol):
    try:
        data = yf.download(symbol, period='5y', interval='1d', progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def prepare_features(data):
    df = data[['Date', 'Close']].copy()
    df = add_all_features(df)
    df['Target'] = df['Close'].shift(-1)
    feature_cols = [c for c in df.columns if c not in ['Date', 'Close', 'Target']]
    for c in feature_cols:
        df[c] = df[c].shift(1)
    df.dropna(inplace=True)
    X = df[feature_cols]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test, feature_cols

def evaluate_on_test_horizons(model, X_test, y_test, horizons):
    metrics = {}
    y_pred_all = model.predict(X_test.values)
    for h, d in horizons.items():
        n = min(d, len(y_test))
        if n < 2:
            metrics[h] = (np.nan, np.nan)
            continue
        true_vals = y_test.values[-n:]
        pred_vals = y_pred_all[-n:]
        metrics[h] = compute_regression_metrics(true_vals, pred_vals)
    return metrics

def fetch_model_results(symbol):
    conn = connect_to_db()
    if not conn:
        return pd.DataFrame()
    try:
        query = """SELECT model_name,
        rmse_1_week, rmse_1_month, rmse_1_year,
        mape_1_week, mape_1_month, mape_1_year
        FROM model_results WHERE stock_symbol = %s;"""
        df = pd.read_sql(query, conn, params=(symbol,))
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

def get_best_models_per_horizon(df):
    horizons = ['1_week', '1_month', '1_year']
    best_models = {}
    for h in horizons:
        col = f"rmse_{h}"
        if col not in df.columns or df[col].isnull().all():
            continue
        idx = df[col].idxmin()
        if pd.isnull(idx):
            continue
        best_models[h] = df.at[idx, 'model_name']
    return best_models

def plot_metric_comparison(df, metric, best_models, bar_width=0.15):
    horizons = ['1_week', '1_month', '1_year']
    colors = {'rmse':'#ff7f0e','mape':'#1f77b4'}
    base_color = colors.get(metric,'#1f77b4')
    model_names = df['model_name']
    indices = list(range(len(model_names)))
    fig, ax = plt.subplots(figsize=(16,8))
    for i,horizon in enumerate(horizons):
        col = f"{metric}_{horizon}"
        vals = pd.to_numeric(df[col], errors='coerce').fillna(0)
        pos = [x + i*bar_width for x in indices]
        colors_list = [base_color if m==best_models.get(horizon) else 'lightgray' for m in model_names]
        bars = ax.bar(pos, vals, width=bar_width, color=colors_list, label=horizon.replace('_',' ').title())
        for bar in bars:
            h = bar.get_height()
            if h>0:
                ax.annotate(f"{h:.3f}", (bar.get_x()+bar.get_width()/2, h), xytext=(0,4), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8)
    ax.set_xticks([r + bar_width*(len(horizons)-1)/2 for r in indices])
    ax.set_xticklabels(model_names, rotation=35, ha='right', fontsize=12)
    title_suffix = " (Lower is Better)"
    ax.set_title(f"{metric.upper()} Metric Comparison Across Horizons{title_suffix}", fontsize=16)
    ax.set_ylabel('Score')
    ax.legend(title='Horizon')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def main():
    st.title("Stocker - Regression Only")
    st.info("Regression models only. Scores saved: RMSE, MAPE for 1w/1m/1y. Best models by lowest RMSE.")

    all_symbols = load_all_nse_symbols()
    selected_symbol = st.selectbox("Select Stock Symbol", all_symbols)

    if not selected_symbol:
        st.info("Please select a stock symbol to proceed.")
        return

    delete_old_model_results(selected_symbol)

    with st.spinner("Fetching stock data..."):
        data = fetch_stock_data(selected_symbol)
    if data is None or data.empty:
        st.warning("No stock data fetched.")
        return

    st.subheader(f"Latesttt data for {selected_symbol}")
    st.dataframe(data.tail(10))

    if len(data) < 100:
        st.warning("At least 100 data points required.")
        return

    X_train, X_test, y_train, y_test, feature_cols = prepare_features(data)

    models = {
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0) if XGBRegressor is not None else None,
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Linear Regression": LinearRegression(),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        "SVM": SVR(kernel='rbf'),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "ANN (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    models = {k: v for k, v in models.items() if v is not None}

    horizons = {'1_week': 5, '1_month': 22, '1_year': 252}  # Removed 5_year
    progress = st.progress(0)
    for i, (name, mdl) in enumerate(models.items()):
        try:
            mdl.fit(X_train.values, y_train.values)
            if not X_test.empty:
                metrics = evaluate_on_test_horizons(mdl, X_test, y_test, horizons)
                save_full_model_result(selected_symbol, 'NSE Index', name, metrics)
        except Exception as e:
            st.error(f"Training failed for {name}: {e}")
        progress.progress((i + 1) / len(models))

    st.success("Regression model training complete. Metrics saved to database (1w/1m/1y only).")

    model_scores_df = fetch_model_results(selected_symbol)
    if model_scores_df.empty:
        st.warning("No model performance data found.")
        return

    best_models = get_best_models_per_horizon(model_scores_df)

    st.subheader("Model Performance Comparison (Regression)")
    with st.expander("Show Metric Comparison Charts"):
        for metric in ['rmse', 'mape']:
            plot_metric_comparison(model_scores_df, metric, best_models)

    st.subheader("Best Model Predictions by Horizon (Regression)")

    hist_data = data[['Date', 'Close']].iloc[-50:].copy()
    hist_data = add_all_features(hist_data)
    hist_data.bfill(inplace=True)

    forecast_days_map = {'1_week': 5, '1_month': 22, '1_year': 252}

    for horizon, model_name in best_models.items():
        if model_name not in models:
            st.warning(f"Model {model_name} not found for horizon {horizon}")
            continue
        model = models[model_name]
        days = forecast_days_map[horizon]
        preds, dates = predict_next_days(model, hist_data, days, feature_cols)
        pred_df = pd.DataFrame({'Date': dates, 'Predicted Close': preds})
        with st.expander(f"{horizon.replace('_', ' ').title()} Prediction — Model: {model_name}"):
            st.dataframe(pred_df)

if __name__ == "__main__":
    main()

