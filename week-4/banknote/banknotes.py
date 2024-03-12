import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from read_data import Reader
from neural_network import BankNoteClassifier

# Read the CSV file and get data
data = Reader("banknotes.csv")
data.read()
X_train, X_test, X_valid, y_train, y_test, y_valid = data.get_split_data() 

# Create an instance of the classifier
model = BankNoteClassifier(input_dim=4, hidden_dim1=7, hidden_dim2=7, output_dim=1)

# Compile the model, also try with optimizer="adam"
model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping. Try different batch_sizes and epochs
model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[early_stopping])

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
