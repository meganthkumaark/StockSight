import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="Nifty 50 Trend Prediction", page_icon="📈", layout="wide")

# Title and description
st.title("Nifty 50 Trend Prediction")
st.markdown("""
This app predicts the next day's trend for the Nifty 50 index (1 = Up, 0 = Down) using a Random Forest model.
Enter the required features below and click **Predict** to see the result.
""")

# Load model and scaler
try:
    rf_model = joblib.load("nifty_trend_rf_model_scaled.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Failed to load model or scaler: {e}")
    st.stop()

# Create input form
st.subheader("Input Features")
with st.form(key="prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        open_price = st.number_input("Open", min_value=-10.0, max_value=10.0, value=0.5, step=0.01, help="Normalized Open price")
        high_price = st.number_input("High", min_value=-10.0, max_value=10.0, value=0.6, step=0.01, help="Normalized High price")
        low_price = st.number_input("Low", min_value=-10.0, max_value=10.0, value=0.4, step=0.01, help="Normalized Low price")
        close_price = st.number_input("Close", min_value=-10.0, max_value=10.0, value=0.55, step=0.01, help="Normalized Close price")
        volume = st.number_input("Volume", min_value=-10.0, max_value=10.0, value=0.3, step=0.01, help="Normalized Volume")
    
    with col2:
        sma_20 = st.number_input("SMA_20", min_value=-10.0, max_value=10.0, value=0.52, step=0.01, help="20-day Simple Moving Average")
        sma_50 = st.number_input("SMA_50", min_value=-10.0, max_value=10.0, value=0.51, step=0.01, help="50-day Simple Moving Average")
        rsi = st.number_input("RSI", min_value=0.0, max_value=100.0, value=60.0, step=0.1, help="Relative Strength Index (0-100)")
        macd = st.number_input("MACD", min_value=-10.0, max_value=10.0, value=0.02, step=0.01, help="MACD line")
        macd_signal = st.number_input("MACD_Signal", min_value=-10.0, max_value=10.0, value=0.01, step=0.01, help="MACD Signal line")
    
    with col3:
        volatility_10d = st.number_input("Volatility_10d", min_value=0.0, max_value=10.0, value=0.05, step=0.01, help="10-day volatility")
        adx = st.number_input("ADX", min_value=0.0, max_value=100.0, value=25.0, step=0.1, help="Average Directional Index (0-100)")
        sentiment_score = st.number_input("Sentiment_Score", min_value=-1.0, max_value=1.0, value=0.7, step=0.01, help="Sentiment Score (-1 to 1)")
        sentiment_momentum = st.number_input("Sentiment_Momentum", min_value=-1.0, max_value=1.0, value=0.1, step=0.01, help="Sentiment Momentum")
        budget_day = st.selectbox("Budget_Day", options=[0, 1], index=0, help="1 if budget day, 0 otherwise")
    
    submit_button = st.form_submit_button(label="Predict")

# Process prediction
if submit_button:
    try:
        # Create DataFrame from inputs
        input_data = pd.DataFrame({
            'Open': [open_price],
            'High': [high_price],
            'Low': [low_price],
            'Close': [close_price],
            'Volume': [volume],
            'SMA_20': [sma_20],
            'SMA_50': [sma_50],
            'RSI': [rsi],
            'MACD': [macd],
            'MACD_Signal': [macd_signal],
            'Volatility_10d': [volatility_10d],
            'ADX': [adx],
            'Sentiment_Score': [sentiment_score],
            'Sentiment_Momentum': [sentiment_momentum],
            'Budget_Day': [budget_day]
        })
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = rf_model.predict(input_scaled)[0]
        probabilities = rf_model.predict_proba(input_scaled)[0]
        
        # Display results
        st.subheader("Prediction Results")
        st.write(f"**Predicted Trend**: {'Up' if prediction == 1 else 'Down'}")
        st.write(f"**Probability of Down (0)**: {probabilities[0]:.2%}")
        st.write(f"**Probability of Up (1)**: {probabilities[1]:.2%}")
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")