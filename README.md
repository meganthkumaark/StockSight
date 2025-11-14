# StockSight
Stock Trend Prediction Using Sentiment Analysis and Technical Indicators
This project predicts daily trends (up/down) for the Nifty 50 index by combining technical indicators and sentiment analysis. Built with Python, it leverages historical data (2010–2023) from Yahoo Finance via yfinance, calculating indicators like SMA, RSI, MACD, and ADX using pandas-ta. Synthetic sentiment scores simulate market mood, with plans for future real-time NLP integration. A Random Forest classifier (scikit-learn) trains on the merged dataset to forecast trends with confidence scores.

The system is deployed as a user-friendly Streamlit web app, allowing users to input market data and sentiment scores to receive predictions. Visualizations (matplotlib, seaborn) compare actual vs. predicted trends, and metrics like accuracy evaluate model performance. Key features include:





Data Collection: Historical Nifty 50 data retrieval.



Feature Engineering: Technical indicators and sentiment scores.



Prediction: Trend forecasts with confidence probabilities.



Deployment: Interactive Streamlit app for accessibility.
