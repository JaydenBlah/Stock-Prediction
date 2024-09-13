import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Set page layout to wide for a more spacious feel
st.set_page_config(layout="wide")

# Title and description
st.title("Stock Price Predictor App")
st.write("""
This application allows you to visualize stock prices, calculate moving averages, and predict future trends using a machine learning model.
""")

# Input for stock symbol
st.title("Stock Price Predictor App")
stock = st.text_input("Enter the Stock ID", "GOOG")

# Fetch stock data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
google_data = yf.download(stock, start, end)

# Load pre-trained model
model = load_model("Latest_stock_price_model.keras")

# Display stock data
st.subheader("Stock Data")
st.dataframe(google_data.tail(10))

# Function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None, title="", labels=("Close Price", "MA")):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(values, 'Orange', label=labels[1])
    ax.plot(full_data.Close, 'b', label=labels[0])
    if extra_data:
        ax.plot(extra_dataset, 'g', label="Additional MA")
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

# Calculate moving averages
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()

# Section: Moving Average Comparisons
st.subheader("Moving Averages for Stock")
st.write("The charts below show the stock's closing prices along with moving averages calculated for different windows.")

# Plots in a two-column layout for professional presentation
col1, col2 = st.columns(2)

# Plot for 250-day MA
with col1:
    st.subheader('MA for 250 Days')
    st.pyplot(plot_graph((10, 6), google_data['MA_for_250_days'], google_data, title="250-Day Moving Average"))

# Plot for 200-day MA
with col2:
    st.subheader('MA for 200 Days')
    st.pyplot(plot_graph((10, 6), google_data['MA_for_200_days'], google_data, title="200-Day Moving Average"))

# Full-width plot for comparison between 100-day and 250-day MAs
st.subheader('Comparing 100-day and 250-day Moving Averages')
st.pyplot(plot_graph((15, 8), google_data['MA_for_100_days'], google_data, extra_data=1, extra_dataset=google_data['MA_for_250_days'], title="100-day and 250-day Moving Averages Comparison"))

# Stock Prediction Section
st.subheader("Stock Price Prediction")

# Prepare data for prediction
splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Generate predictions
predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Prepare dataframe for plotting results
ploting_data = pd.DataFrame({
    'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
}, index=google_data.index[splitting_len + 100:])

# Display original and predicted values
st.subheader("Original vs Predicted Stock Prices")
st.dataframe(ploting_data.tail(10))

# Plot original vs predicted data
fig = plt.figure(figsize=(15, 10))
plt.plot(pd.concat([google_data.Close[:splitting_len + 100], ploting_data], axis=0))
plt.legend(["Data Not Used", "Original Test Data", "Predicted Test Data"], loc="upper left")
plt.title("Stock Price Prediction", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig)

# Footer
st.write("Designed with 💻 by Jayden")
