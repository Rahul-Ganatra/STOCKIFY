from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
from model import prepare_data, build_model, predict_future_prices
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__, 
    template_folder='./',  # Look for templates in current directory
    static_folder='./'     # Look for static files in current directory
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data['symbol']
        years = int(data['years'])
        
        # Calculate dates
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
        
        # Fetch stock data
        print(f"Fetching data for {symbol}...")
        df = yf.download(symbol, start=start_date, end=end_date)
        
        if df.empty:
            return jsonify({'error': 'No data found for the given symbol'})
        
        # Prepare data and make predictions
        X_train, y_train, X_test, y_test, scaler = prepare_data(df)
        model = build_model(60)  # sequence_length = 60
        
        print("Training model...")
        model.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.1, verbose=1)
        
        # Make predictions
        last_sequence = X_test[-1]
        future_pred = predict_future_prices(model, last_sequence, scaler)
        
        # Create future dates
        future_dates = pd.date_range(start=pd.Timestamp.today(), periods=31)[1:]
        
        # Prepare prediction results
        predictions = []
        for date, price in zip(future_dates, future_pred):
            predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': float(price[0])
            })
        
        # Create separate plots for 30-day and historical views
        def create_plot():
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            return base64.b64encode(img.getvalue()).decode()

        # Create 30-day plot
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[-30:], df['Close'].values[-30:], label='Historical Data')
        plt.plot(future_dates, future_pred, '--', label='Predicted Prices')
        plt.title(f'{symbol} Stock Price Prediction (Last 30 Days)')
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        thirty_day_plot = create_plot()
        plt.close()

        # Create historical plot
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'].values, label=f'Historical Data ({years} Years)')
        plt.plot(future_dates, future_pred, '--', label='Predicted Prices')
        plt.title(f'{symbol} Stock Price Prediction (Full Historical Data)')
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        historical_plot = create_plot()
        plt.close()

        return jsonify({
            'success': True,
            'predictions': predictions,
            'thirtyDayPlot': thirty_day_plot,
            'historicalPlot': historical_plot
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/market_indices')
def get_market_indices():
    try:
        # Fetch data for market indices using yfinance
        nifty50 = yf.Ticker("^NSEI")
        sensex = yf.Ticker("^BSESN")
        niftybank = yf.Ticker("^NSEBANK")

        # Get current data
        nifty50_data = nifty50.history(period='1d')
        sensex_data = sensex.history(period='1d')
        niftybank_data = niftybank.history(period='1d')

        # Calculate percentage changes
        nifty50_change = ((nifty50_data['Close'].iloc[-1] - nifty50_data['Open'].iloc[-1]) / 
                         nifty50_data['Open'].iloc[-1] * 100)
        sensex_change = ((sensex_data['Close'].iloc[-1] - sensex_data['Open'].iloc[-1]) / 
                        sensex_data['Open'].iloc[-1] * 100)
        niftybank_change = ((niftybank_data['Close'].iloc[-1] - niftybank_data['Open'].iloc[-1]) / 
                           niftybank_data['Open'].iloc[-1] * 100)

        return jsonify({
            'success': True,
            'nifty50': {
                'value': round(float(nifty50_data['Close'].iloc[-1]), 2),
                'change': round(float(nifty50_change), 2)
            },
            'sensex': {
                'value': round(float(sensex_data['Close'].iloc[-1]), 2),
                'change': round(float(sensex_change), 2)
            },
            'niftybank': {
                'value': round(float(niftybank_data['Close'].iloc[-1]), 2),
                'change': round(float(niftybank_change), 2)
            }
        })
    except Exception as e:
        print(f"Error in market_indices: {str(e)}")  # Add debugging
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
