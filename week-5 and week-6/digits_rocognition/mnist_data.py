from tf_config import tf

class MNISTData:
	def __init__(self):
		# Use MNIST handwriting dataset
		mnist = tf.keras.datasets.mnist

		# Prepare data for training
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		# Normalizing the pixel values to a range between 0 and 1
		x_train, x_test = x_train / 255.0, x_test / 255.0

		# convert class labels into one-hot encoded vectors
		self.y_train = tf.keras.utils.to_categorical(y_train)
		self.y_test = tf.keras.utils.to_categorical(y_test)

		# reshape the input image data
		self.x_train = x_train.reshape(
    		x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
		)
		self.x_test = x_test.reshape(
    		x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
		)

	def print_shape(self):
		print(f"Training data, x shape: {self.x_train.shape}")
		print(f"Training data, y shape: {self.y_train.shape}")
		print(f"Testing data, x shape: {self.x_test.shape}")
		print(f"Testing data, y shape: {self.y_test.shape}")
