from read_data import Reader
from nn_lstm import ETHPricePredict

sequence_length = 10 # adjust the sequence length as needed

# Get data
data = Reader(sequence_length)

train_generator, test_generator = data.train_test_generator()

# Create an instance of the model
model = ETHPricePredict(sequence_length)

# Compile and build the model
model.compile(optimizer='adam', loss='mean_squared_error')
#model.build((None, sequence_length, 1))
#model.summary()

# Train the model using generator
model.fit(train_generator, epochs=10)

# Predictions using generator
predicted_values = model.predict(test_generator)

# Generate predictions for the next 10 days
# And plot the histocal + predicted trends
data.plot_prediction(model, predicted_values)

