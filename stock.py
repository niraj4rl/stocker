import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
from datetime import timedelta

# -------------------- Load NSE symbols --------------------
@st.cache_data(show_spinner=False)
def load_all_nse_symbols():
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbol_col = 'SYMBOL' if 'SYMBOL' in df.columns else df.columns[0]
        symbols = df[symbol_col].str.strip().tolist()
        symbols = [sym + ".NS" for sym in symbols]
        return symbols
    except Exception as e:
        st.error(f"Failed to fetch NSE symbols: {e}")
        return []

# -------------------- Fetch stock data --------------------
@st.cache_data(show_spinner=False)
def fetch_stock_data(symbol, period='1y', interval='1d'):
    try:
        data = yf.download(symbol, period=period, interval=interval)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# -------------------- Prepare features --------------------
def prepare_features(data):
    data['Target'] = data['Close'].shift(-1)
    df = data[['Close', 'Target']].dropna()
    X = df[['Close']]
    y = df['Target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- Train and evaluate --------------------
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, preds)
    accuracy = 100 - mape * 100
    return model, preds, rmse, accuracy

# -------------------- Predict next days --------------------
def predict_next_days(model, last_close, last_date, n_days=5):
    preds = []
    dates = []
    value = last_close
    date = last_date
    for i in range(1, n_days + 1):
        # FIX: reshape correctly for sklearn
        value = model.predict(np.array(value).reshape(1, -1))[0]
        date = date + timedelta(days=1)

        # Skip weekends
        while date.weekday() >= 5:
            date += timedelta(days=1)

        preds.append(value)
        dates.append(date)

    return pd.DataFrame({'Date': dates, 'Predicted Close': preds})

# -------------------- Main --------------------
def main():
    st.title("model")

    with st.spinner("Loading all NSE stock symbols..."):
        symbols = load_all_nse_symbols()

    if symbols:
        st.success(f"Loaded {len(symbols)} symbols")
        selected = st.selectbox("Select a stock symbol", symbols)

        if selected:
            with st.spinner(f"Downloading data for {selected}..."):
                data = fetch_stock_data(selected)

            if data is not None and not data.empty:
                st.subheader(f"Latest data for {selected}")
                st.dataframe(data.tail())

                X_train, X_test, y_train, y_test = prepare_features(data)
                model, preds, rmse, accuracy = train_and_evaluate(
                    X_train, X_test, y_train, y_test
                )

                st.success("Model Trained Successfully")
                st.write(f"RMSE: {rmse:.4f}")
                st.write(f"Prediction Accuracy (based on MAPE): {accuracy:.2f}%")

                st.subheader("Sample predictions on test set:")
                st.write(preds[:5])

                last_close = data['Close'].iloc[-1]
                last_date = data['Date'].iloc[-1]
                if isinstance(last_date, str):
                    last_date = pd.to_datetime(last_date)

                future_pred_df = predict_next_days(model, last_close, last_date, n_days=5)

                st.subheader("Next 5 day predictions with dates:")
                st.dataframe(future_pred_df)

            else:
                st.warning("No data found for this symbol.")
    else:
        st.error("Could not load any NSE symbols.")

if _name_ == "_main_":
    main()

