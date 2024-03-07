import tensorflow as tf
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	

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

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(0.002), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

