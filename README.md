# 📈 StockPulse - Intelligent Stock Market Forecasting Dashboard
## 🚀 Overview
**StockPulse** is an interactive stock market analytics and forecasting platform built with **Streamlit**, designed to help investors, traders, and data enthusiasts analyze historical stock data and generate future price forecasts using advanced machine learning and time-series forecasting models.
The application fetches real-time historical stock data directly from Yahoo Finance and provides comprehensive visualizations, stationarity testing, time-series decomposition, and forecasting using multiple predictive models.
## ✨ Features
### 📊 Real-Time Stock Data
* Fetches historical stock prices directly from Yahoo Finance.
* Supports multiple major technology and growth stocks including:

  * Apple (AAPL)
  * Microsoft (MSFT)
  * Amazon (AMZN)
  * Nvidia (NVDA)
  * Tesla (TSLA)
  * Meta (META)
  * Netflix (NFLX)
  * Google (GOOG/GOOGL)
  * AMD
  * Qualcomm (QCOM)
  * And more.

### 📈 Interactive Visualizations

* Dynamic stock price charts using Plotly.
* Zoom, pan, and customize date ranges.
* Explore Open, High, Low, Close, Volume, and Adjusted Close values.

### 🔍 Time Series Analysis

* Augmented Dickey-Fuller (ADF) Stationarity Test.
* Seasonal Decomposition:

  * Trend
  * Seasonality
  * Residual Components

### 🤖 Forecasting Models

#### 1. SARIMA (Seasonal ARIMA)

* Adjustable parameters:

  * p (autoregressive order)
  * d (differencing order)
  * q (moving average order)
  * seasonal period
* Model summary and diagnostics.
* Future stock price forecasting.

#### 2. Random Forest Regression

* Machine learning-based prediction.
* Automatic train-test split.
* Performance evaluation using RMSE.

#### 3. Facebook Prophet

* Trend and seasonality-aware forecasting.
* Long-term future predictions.
* Automatic handling of missing data and outliers.

### 📉 Performance Evaluation

* Root Mean Squared Error (RMSE)
* Actual vs Predicted visual comparison

---

## 🛠️ Tech Stack

| Category             | Technologies                |
| -------------------- | --------------------------- |
| Frontend             | Streamlit                   |
| Data Processing      | Pandas, NumPy               |
| Visualization        | Plotly, Matplotlib, Seaborn |
| Data Source          | Yahoo Finance (yFinance)    |
| Statistical Analysis | Statsmodels                 |
| Forecasting          | Prophet, SARIMA             |
| Machine Learning     | Scikit-Learn                |


```txt
streamlit
pandas
numpy
matplotlib
seaborn
plotly
yfinance
statsmodels
prophet
scikit-learn
```



## 📊 Workflow

1. Select a stock ticker.
2. Choose start and end dates.
3. Explore historical stock data.
4. Visualize trends and patterns.
5. Check stationarity using ADF Test.
6. Analyze trend, seasonality, and residual components.
7. Select a forecasting model:

   * SARIMA
   * Random Forest
   * Prophet
8. Generate forecasts and compare predictions.

---

## 📸 Application Preview

### Dashboard

* Historical Stock Data
* Interactive Price Charts
* Statistical Analysis

### Forecasting

* SARIMA Predictions
* Random Forest Regression
* Prophet Future Forecasts

---

## 🎯 Use Cases

* Stock Market Analysis
* Financial Data Exploration
* Time Series Forecasting
* Machine Learning Demonstrations
* Educational Projects
* Portfolio Research

---

## 🔮 Future Enhancements

* Support for Cryptocurrency Forecasting
* LSTM Deep Learning Models
* XGBoost Forecasting
* Technical Indicators

  * RSI
  * MACD
  * Bollinger Bands
* Portfolio Optimization
* Model Comparison Dashboard
* Forecast Accuracy Metrics Dashboard
* News Sentiment Analysis



