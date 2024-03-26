from tf_config import tf
import sys

from mnist_data import MNISTData
from cnn_model import DigitClassifier

data = MNISTData()
data.print_shape()

digits_classifier = DigitClassifier()
digits_classifier.build((None, 28, 28, 1))
digits_classifier.summary()

# Train neural network
digits_classifier.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

digits_classifier.fit(data.x_train, data.y_train, epochs=20)

# Evaluate neural network performance
digits_classifier.evaluate(data.x_test, data.y_test, verbose=2)

# Save model to file
filename = "model.tf"
digits_classifier.save(filename, save_format="tf")
print(f"Model saved to {filename}.")

