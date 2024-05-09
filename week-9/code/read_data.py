import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Reader:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length

        # Downloading Ethereum Data From Yahoo Finance
        self.eth_data = yf.download('ETH-USD', start='2018-01-01', end='2021-01-01')
        self.preprocessing()

    def preprocessing(self):
        # Processing the data
        data = self.eth_data['Close'].values.reshape(-1, 1)
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaled_data = self.scaler.fit_transform(data)

    def train_test_generator(self):
       # Create TimeseriesGenerator for train and test sets
       train_generator = TimeseriesGenerator(self.scaled_data, self.scaled_data, length=self.sequence_length, batch_size=1)
       test_generator = TimeseriesGenerator(self.scaled_data, self.scaled_data, length=self.sequence_length, batch_size=1)

       return train_generator, test_generator

    def plot_prediction(self, model, predicted_values):
       # Inverse transform predictions to original scale
       predicted_values = self.scaler.inverse_transform(predicted_values)

       # Get the last sequence from the training data
       last_sequence = self.scaled_data[-self.sequence_length:]

       # Generate predictions for the next 10 days
       predictions = []

       for _ in range(10):
           current_prediction = model.predict(last_sequence.reshape(1, self.sequence_length, 1))[0, 0]
           predictions.append(current_prediction)
           last_sequence = np.append(last_sequence[1:], current_prediction).reshape(-1, 1)

       # Inverse transform predictions to original scale
       predicted_values = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
       # Generate dates for the next 10 days
       last_date = self.eth_data.index[-1]
       dates = pd.date_range(start=last_date, periods=11)[1:]

       # 11 because the first date is already in the data
       # Plotting the predicted values for the next 10 days
       plt.figure(figsize=(10, 6))
       plt.plot(self.eth_data.index, self.eth_data['Close'], label='Historical Data')
       plt.plot(dates, predicted_values, label='Predicted')
       plt.xlabel('Date')
       plt.ylabel('ETH Price')
       plt.title('Predicted Ethereum Prices for the Next 10 Days')
       plt.legend()
       plt.grid(True)
       plt.show()


