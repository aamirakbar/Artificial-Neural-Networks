import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Dropout

# Create a neural network
class BankNoteClassifier(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(BankNoteClassifier, self).__init__()
        self.input_layer = InputLayer(input_shape=(input_dim,))
        self.dense1 = Dense(hidden_dim1, activation="relu")
        self.dropout1 = Dropout(0.2)  # Adding dropout after the first dense layer
        self.dense2 = Dense(hidden_dim2, activation="relu")
        self.dropout2 = Dropout(0.2)  # Adding dropout after the second dense layer
        self.dense3 = Dense(output_dim, activation="tanh")

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.dense1(x)
        x = self.dropout1(x)  # Applying dropout after the first dense layer
        x = self.dense2(x)
        x = self.dropout2(x)  # Applying dropout after the second dense layer
        return self.dense3(x)