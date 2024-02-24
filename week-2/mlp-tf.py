import tensorflow as tf
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense

# Define the model
class MLP(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.input_layer = InputLayer(input_shape=(input_dim,))
        self.dense1 = Dense(hidden_dim1, activation="relu")
        self.dense2 = Dense(hidden_dim2, activation="relu")
        self.dense3 = Dense(output_dim,  activation="tanh")

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

# Create an instance of the model
model = MLP(input_dim=5, hidden_dim1=7, hidden_dim2=7, output_dim=1)

# Print model summary
model.build((None, 5))
model.summary()

