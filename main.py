import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from mplcursors import cursor
from matplotlib.widgets import Button, Slider, CheckButtons

def get_stock_data(symbol, start_date, end_date):
    """
    Fetch stock data using yfinance API
    """
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def prepare_data(df, sequence_length=60):
    """
    Prepare data for LSTM model
    """
    # Use closing price
    data = df['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler

def build_model(sequence_length):
    """
    Create LSTM model
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_prices(model, last_sequence, scaler, num_days=30):
    """
    Predict future stock prices
    """
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(num_days):
        # Get prediction for next day
        next_day_pred = model.predict(current_sequence.reshape(1, -1, 1))
        future_predictions.append(next_day_pred[0, 0])
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_day_pred
        
    # Inverse transform predictions
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return future_predictions

def plot_interactive(df, future_dates, future_pred, symbol, years):
    """
    Create interactive plot with controls
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.subplots_adjust(bottom=0.25)  # Make room for controls
    
    # Get last 30 days of historical data
    last_30_days = df.index[-30:]
    last_30_days_prices = df['Close'].values[-30:]
    
    # Create the plots
    historical_line = ax.plot(last_30_days, last_30_days_prices, 
                            label='Historical Data', color='blue')[0]
    prediction_line = ax.plot(future_dates, future_pred, 
                            label='Predicted Prices', linestyle='--', color='red')[0]
    
    ax.set_title(f'{symbol} Stock Price Prediction\n{years} Year(s) Analysis')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (₹)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Set initial y-axis limits with padding
    all_prices = np.concatenate([last_30_days_prices, future_pred.flatten()])
    ax.set_ylim(min(all_prices) * 0.95, max(all_prices) * 1.05)
    
    # Add interactive cursors
    cursor_historical = cursor(historical_line)
    cursor_prediction = cursor(prediction_line)
    
    @cursor_historical.connect('add')
    def on_add(sel):
        date = last_30_days[int(sel.target.index)]
        price = last_30_days_prices[int(sel.target.index)]
        sel.annotation.set_text(f'Date: {date.strftime("%Y-%m-%d")}\nPrice: ₹{price:.2f}')

    @cursor_prediction.connect('add')
    def on_add(sel):
        date = future_dates[int(sel.target.index)]
        price = future_pred[int(sel.target.index)][0]
        sel.annotation.set_text(f'Date: {date.strftime("%Y-%m-%d")}\nPredicted: ₹{price:.2f}')
    
    # Add control buttons
    ax_reset = plt.axes([0.8, 0.05, 0.1, 0.04])
    button_reset = Button(ax_reset, 'Reset View')
    
    ax_timeframe = plt.axes([0.1, 0.05, 0.2, 0.04])
    check_timeframe = CheckButtons(ax_timeframe, ['Show Full History'], [False])
    
    # Replace slider with zoom in/out buttons
    ax_zoomin = plt.axes([0.35, 0.05, 0.1, 0.04])
    ax_zoomout = plt.axes([0.5, 0.05, 0.1, 0.04])
    button_zoomin = Button(ax_zoomin, 'Zoom In')
    button_zoomout = Button(ax_zoomout, 'Zoom Out')
    
    # Store original view limits
    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()
    
    # Initialize with 30-day view
    historical_line.set_data(last_30_days, last_30_days_prices)
    ax.set_xlim(last_30_days[0], future_dates[-1])
    plt.draw()
    
    def reset(event):
        if check_timeframe.get_status()[0]:  # If showing full history
            ax.set_xlim(df.index[0], future_dates[-1])
        else:  # If showing 30-day view
            ax.set_xlim(last_30_days[0], future_dates[-1])
        ax.set_ylim(original_ylim)
        plt.draw()
    
    def toggle_timeframe(label):
        if check_timeframe.get_status()[0]:  # If checked
            # Show historical data based on user-selected years
            historical_start = df.index[-min(len(df), 252 * years)]  # 252 trading days per year
            ax.set_xlim(historical_start, future_dates[-1])
            historical_line.set_data(df.index, df['Close'].values)
        else:
            # Show last 30 days
            ax.set_xlim(last_30_days[0], future_dates[-1])
            historical_line.set_data(last_30_days, last_30_days_prices)
        plt.draw()
    
    def zoom_in(event):
        # Zoom in by 20%
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        xcenter = np.mean(cur_xlim)
        ycenter = np.mean(cur_ylim)
        
        ax.set_xlim([xcenter - (cur_xlim[1] - cur_xlim[0]) * 0.4,
                    xcenter + (cur_xlim[1] - cur_xlim[0]) * 0.4])
        ax.set_ylim([ycenter - (cur_ylim[1] - cur_ylim[0]) * 0.4,
                    ycenter + (cur_ylim[1] - cur_ylim[0]) * 0.4])
        plt.draw()
    
    def zoom_out(event):
        # Zoom out by 20%
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        xcenter = np.mean(cur_xlim)
        ycenter = np.mean(cur_ylim)
        
        ax.set_xlim([xcenter - (cur_xlim[1] - cur_xlim[0]) * 0.6,
                    xcenter + (cur_xlim[1] - cur_xlim[0]) * 0.6])
        ax.set_ylim([ycenter - (cur_ylim[1] - cur_ylim[0]) * 0.6,
                    ycenter + (cur_ylim[1] - cur_ylim[0]) * 0.6])
        plt.draw()
    
    button_reset.on_clicked(reset)
    check_timeframe.on_clicked(toggle_timeframe)
    button_zoomin.on_clicked(zoom_in)
    button_zoomout.on_clicked(zoom_out)
    
    # Enable pan and zoom with mouse
    plt.gcf().canvas.toolbar.pan()
    
    plt.show()

# Get user inputs
print("\nEnter the stock symbol with exchange suffix (e.g., RELIANCE.NS, TCS.NS):")
symbol = input().upper()

print("\nEnter the number of years for analysis (1-10):")
while True:
    try:
        years = int(input())
        if 1 <= years <= 10:
            start_date = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
            break
        else:
            print("Please enter a number between 1 and 10")
    except ValueError:
        print("Please enter a valid number")

end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
sequence_length = 60
future_days = 30

print(f"\nFetching {years} year(s) of data for {symbol}...")

# Get data
df = get_stock_data(symbol, start_date, end_date)

if df is not None:
    # Prepare data
    X_train, y_train, X_test, y_test, scaler = prepare_data(df, sequence_length)
    
    # Build and train model
    model = build_model(sequence_length)
    history = model.fit(
        X_train, 
        y_train, 
        batch_size=32, 
        epochs=20, 
        validation_split=0.1,
        verbose=1
    )
    
    # Make future predictions
    last_sequence = X_test[-1]
    future_pred = predict_future_prices(model, last_sequence, scaler, future_days)
    
    # Create future dates (starting from tomorrow)
    last_date = pd.Timestamp.today()
    future_dates = pd.date_range(start=last_date, periods=future_days+1)[1:]
    
    # Create interactive plot
    plot_interactive(df, future_dates, future_pred, symbol, years)
    
    # Print predicted prices
    print("\nPredicted prices for the next 30 days:")
    for date, price in zip(future_dates, future_pred):
        print(f"{date.date()}: ₹{price[0]:.2f}")
