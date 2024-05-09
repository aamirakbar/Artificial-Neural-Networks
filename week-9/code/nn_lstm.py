import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense

# Create a neural network
class ETHPricePredict(tf.keras.Model):
    def __init__(self, sequence_length):
        super(ETHPricePredict, self).__init__()

        # 50 units, meaning it will have 50 memory cells to process input sequences and capture temporal dependencies
        # return_sequences=True indicates that this layer should return sequences of hidden states for each time step in the input sequence because the next layer is another LSTM layer, requiring sequence data as input.
        self.input_lstm_layer = LSTM(50, return_sequences=True, input_shape=(sequence_length, 1))

        # return_sequences is set to False, meaning this layer will only return the output at the last time step.
        self.lstm_layer = LSTM(50)
        self.output_layer = Dense(1)

    def call(self, inputs):
        h_t = self.input_lstm_layer(inputs) #h_t shape = (batch_size, sequence_length, units).
        h_last = self.lstm_layer(h_t)  # h_last shape = (batch_size, units)
        return self.output_layer(h_last)
