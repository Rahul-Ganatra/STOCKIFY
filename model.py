import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def prepare_data(df, sequence_length=60):
    """
    Prepare data for LSTM model
    """
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
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
        next_day_pred = model.predict(current_sequence.reshape(1, -1, 1))
        future_predictions.append(next_day_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_day_pred
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return future_predictions
