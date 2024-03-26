from tf_config import tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class DigitClassifier(tf.keras.Model):
	def __init__(self):
		super(DigitClassifier, self).__init__()
		# Convolutional layer. Learn 32 filters using a 3x3 kernel
		self.conv_layer = Conv2D(
        	32, (3, 3), activation="relu", input_shape=(28, 28, 1)
    	)
		
		# Max-pooling layer, using 2x2 pool size
		self.max_pool_layer = MaxPooling2D(pool_size=(2, 2))

		# Flatten units
		self.flatten_layer = Flatten()

		# Add hidden layer1 with dropout
		self.hidden_layer = Dense(128, activation="relu")
		self.dropout_layer = Dropout(0.5)

		# Add an output layer with output units for all 10 digits
		self.output_layer = Dense(10, activation="softmax")

	def call(self, inputs):
		x = self.conv_layer(inputs)
		x = self.max_pool_layer(x)
		x = self.flatten_layer(x)
		x = self.hidden_layer(x)
		x = self.dropout_layer(x)
		return self.output_layer(x)
		
		
