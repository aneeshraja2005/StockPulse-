import streamlit as st
import pandas as pd,numpy as np,matplotlib.pyplot as plt , seaborn as sns, plotly.express as px
import plotly.graph_objects as go
import yfinance as yf 
from datetime import date,timedelta
import  statsmodels.api as sm
from  statsmodels.tsa.seasonal import seasonal_decompose
from  statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

app_name ="StockPulse"
st.title(app_name)
st.subheader("Your intelligent companion for stock market forecasting.")
st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ2Ub1IABTKL3VoRh6fOFZkY0oOU35KEbhn9w&s")
st.sidebar.header("Forecast Settings")
start_date = st.sidebar.date_input('Start date', date(2022, 1,1))
end_date = st.sidebar.date_input('End date', date(2026, 5,5 ))
ticker_list = [ "AMZN",   "META", "TSLA", "NVDA","AAPL", "ADBE", "PYPL", "GOOG","INTC", "CMCSA", "NFLX","MSFT", "PEP",  "AVGO", "COST","GOOGL", "AMD", "QCOM"]
ticker = st.sidebar.selectbox("Select Company for Analysis", ticker_list)
df = yf.download(ticker, start=start_date, end=end_date)
df.columns=df.columns.droplevel(level=1)
df.insert(0,"Date",df.index,True)
df.reset_index(drop=True,inplace=True)
st.write('Data from', start_date, 'to', end_date)
st.write(df)
st.header("Data Visualization")
st.subheader("Visual Representation of Stock Data")
st.write("**Tip:** Select a date range from the sidebar or zoom into the chart.")
fig = px.line(df, x='Date', y=df.columns, title='Closing price of the stock', width=1000, height=600)
st.plotly_chart(fig)
column = st.selectbox("Choose feature for forecasting", df.columns[1:])
df = df[['Date', column]]
st.write("Selected Data")
st.write(df)
st.header('Is data Stationary?')
st.write(adfuller(df[column])[1] < 0.05)
st.header('Decomposition of the data')
decomposition = seasonal_decompose(df[column], model='additive', period=12)
st.write(decomposition.plot())
st.write("## Plotting the decomposition ")
st.plotly_chart(px.line(x=df["Date"], y=decomposition.trend, title='Trend', width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='red'))
st.plotly_chart(px.line(x=df["Date"], y=decomposition.seasonal, title='Seasonality', width=1000, height=400,
labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=df["Date"], y=decomposition.resid, title='Residuals', width=1000, height=400,
labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue', line_dash='dot'))
models = ['SARIMA', 'Random Forest', 'Prophet']
selected_model=st.sidebar.selectbox('Select the model for forecasting', models)

if selected_model == 'SARIMA':
    p = st.slider('Select the value of p', 0, 5, 2)
    d = st.slider('Select the value of d', 0, 5, 1)
    q = st.slider('Select the value of q', 0, 5, 2)
    seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 12)
    model = sm.tsa.statespace.SARIMAX(df[column], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
    model = model.fit()
    st.header('Model Summary')
    st.write(model.summary())
    st.write("---")
    st.write("<p style='color:green; font-size: 50px; font-weight: bold;'>Forecasting the data with SARIMA</p>",
             unsafe_allow_html=True)
    forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 10)
    predictions = model.get_prediction(start=len(df), end=len(df) + forecast_period)
    predictions = predictions.predicted_mean
    predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
    predictions = pd.DataFrame(predictions)
    predictions.insert(0, "Date", predictions.index, True)
    predictions.reset_index(drop=True, inplace=True)
    st.write("Predictions", predictions)
    st.write("Actual Data", df)
    st.write("---")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df[column], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted',
                   line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
    st.plotly_chart(fig)
elif selected_model == 'Random Forest':
   
    st.header('Random Forest Regression')
    train_size = int(len(df) * 0.8)
    train_data, test_data = df[:train_size], df[train_size:]
    train_X, train_y = train_data['Date'], train_data[column]
    test_X, test_y = test_data['Date'], test_data[column]
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_model.fit(train_X.values.reshape(-1, 1), train_y.values)
    predictions = rf_model.predict(test_X.values.reshape(-1, 1))
    mse = mean_squared_error(test_y, predictions)
    rmse = np.sqrt(mse)
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    combined_data = pd.concat([train_data, test_data])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined_data["Date"], y=combined_data[column], mode='lines', name='Actual',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data["Date"], y=predictions, mode='lines', name='Predicted',
                             line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted (Random Forest)', xaxis_title='Date', yaxis_title='Price',
                      width=1000, height=400)
    st.plotly_chart(fig)
elif selected_model == 'Prophet':
   
    st.header('Facebook Prophet')
    prophet_data = df[['Date', column]]
    prophet_data = prophet_data.rename(columns={'Date': 'ds', column: 'y'})
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)
    future = prophet_model.make_future_dataframe(periods=365)
    forecast = prophet_model.predict(future)
    fig = prophet_model.plot(forecast)
    plt.title('Forecast with Facebook Prophet')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig)

st.write("Model selected:", selected_model)



