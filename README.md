# Stocker – NSE Stock Price Prediction

**June 2025 - Ongoing (Final Year Project)**  
End-to-end Streamlit web app for NSE stock forecasting - currently under active development with multiple ML models, PostgreSQL metrics storage, and horizon-optimized predictions spanning 1-week to 5-year horizons using RMSE, MAPE, F1, and Cohen's Kappa evaluation.

## Problem Statement
- Traditional stock prediction fails across **varying time horizons** - trees excel short-term momentum, linears survive long-term drift
- Investors need **symbol+horizon specific model recommendations** not generic ensembles
- Nifty 50 lacks comprehensive **multi-metric benchmarks** combining price error + directional accuracy
- Single-model apps ignore performance degradation patterns critical for practical deployment

## System Architecture
Modular pipeline flows from Streamlit UI through ML backend to data and persistence layers. User selects Nifty 50 stock triggering cached yfinance fetch, automated feature engineering with 20+ technical indicators, parallel training of 8 regression models, multi-horizon metric computation, PostgreSQL persistence, averaged-rank model selection, and autoregressive forecasting. Caching dramatically accelerates retraining while database enables cross-session leaderboards and performance evolution tracking.

## Data Pipeline
- Live NSE equities loaded from official CSV (2000+ valid `.NS` symbols)
- 5-year daily OHLCV via `yfinance(period='5y', interval='1d')`
- Extracts Date/Close columns, validates minimum 100 points
- Chronological 80/20 train-test split preserves time-series integrity
- Recent 10-day data preview confirms quality before training

## Feature Engineering
Temporal, momentum, and technical features engineered avoiding leakage:
- **Temporal**: dayofweek, month, dayofyear, weekofyear seasonality
- **Momentum**: Close lags 1-5 days + 7-day rolling mean/standard deviation
- **RSI(14)**: 14-day gain/loss ratio for overbought/oversold detection
- **MACD**: EMA12-EMA26 difference + 9-period signal line
- **Bollinger Bands**: SMA20 ± 2×std20 with bandwidth for volatility squeezes
All features shifted once before target creation (next Close) yielding ~1200 clean samples.

## Modeling Approach
Eight benchmark regression models trained with fixed hyperparameters for fair comparison:
- **Tree Ensembles**: XGBoost(100), RandomForest(100), AdaBoost(100)
- **Instance-Based**: SVR(RBF), KNN(k=5)
- **Parametric**: LinearRegression, DecisionTree(depth=6)
- **Neural**: MLP(64→32 hidden, 500 iterations)
XGBoost gracefully falls back if unavailable. Models fit chronologically ordered training data predicting next-day closing price.

## Multi-Horizon Strategy
**Realistic investor timeframes** tested via sliding windows on chronological test set:
- 1-week: last 5 trading days (momentum dominant)
- 1-month: last 22 days (trend persistence)
- 1-year: last 252 days (mean reversion begins)
Metrics computed per window reveal characteristic degradation patterns across horizons.

## Model Selection Methodology
**Averaged ranking across four metrics per horizon** ensures robust selection:
1. Individual ranks: min(RMSE, MAPE), max(F1, Kappa)
2. Mean rank calculation across all available metrics
3. **Lowest average rank wins** per horizon
**Example results**: Random Forest dominates 1-week, XGBoost 1-month, SVM survives 1-year
Bar charts highlight best models in gold with 3-decimal precision annotations.

## Database & Persistence Layer
PostgreSQL stores comprehensive `model_results` table capturing:
- 16 metrics per stock/index/model/horizon combination
- Auto-deletes prior evaluations per symbol ensuring fresh results
- Safe float conversion handles NaN/infinity values
- Pandas `read_sql()` queries generate instant leaderboards
Enables cross-session performance comparison and model evolution tracking.

**Current Status**: Active development by team of 4 - targeting LSTM integration, backtesting framework, hyperparameter optimization, and paper publication by project completion.
