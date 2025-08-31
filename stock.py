import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

@st.cache_data(show_spinner=False)
def load_all_nse_symbols():
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        # Adjust this if the column name differs
        symbol_col = 'SYMBOL' if 'SYMBOL' in df.columns else df.columns[0]
        symbols = df[symbol_col].str.strip().tolist()
        symbols = [sym + ".NS" for sym in symbols]
        return symbols
    except Exception as e:
        st.error(f"Failed to fetch NSE symbols: {e}")
        return []

@st.cache_data(show_spinner=False)
def fetch_stock_data(symbol, period='1y', interval='1d'):
    try:
        data = yf.download(symbol, period=period, interval=interval)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def prepare_features(data):
    data['Target'] = data['Close'].shift(-1)
    df = data[['Close', 'Target']].dropna()
    X = df[['Close']]
    y = df['Target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    return model, preds, rmse

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
                st.write(f"Latest data for {selected}")
                st.dataframe(data.tail())

                X_train, X_test, y_train, y_test = prepare_features(data)
                model, preds, rmse = train_and_evaluate(X_train, X_test, y_train, y_test)

                st.success(f" RMSE: {rmse:.4f}")
                st.write("next 5 day predictions:", preds[:5])
            else:
                st.warning("No data found for this symbol.")
    else:
        st.error("Could not load any NSE symbols.")

if __name__ == "__main__":
    main()


