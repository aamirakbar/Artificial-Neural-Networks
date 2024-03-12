import tensorflow as tf
from neural_network import BankNoteClassifier

# Create an instance of the model
model = BankNoteClassifier(input_dim=4, hidden_dim1=7, hidden_dim2=7, output_dim=1)

# Print model summary
model.build((None, 4))
model.summary()
