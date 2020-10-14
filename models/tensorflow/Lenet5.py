class Lenet5(object):
	def get_model(self):
		"""
		Lenet5 expects 32 x 32 image size. 

		Model taken from https://engmrk.com/lenet-5-a-classic-cnn-architecture/
		"""
		import tensorflow as tf
		from tensorflow.keras import layers
		
		model  = tf.keras.Sequential()
		
		model.add(tf.keras.Input(shape = (32, 32, 3,)))
		
		model.add(layers.Conv2D(
			filters=6, kernel_size=(5,5), 
			strides=(1,1), activation="relu")
		)

		model.add(layers.AveragePooling2D(
			pool_size = (2,2),) 
		)
		
		model.add(layers.Conv2D(
			filters=16, kernel_size=(5,5), 
			strides=(1,1), activation="relu")
		)

		model.add(layers.Conv2D(
			filters=6, kernel_size=(2,2), 
			strides=(2,2), activation="relu")
		)

		model.add(layers.Flatten())
		model.add(layers.Dense(units = 120))
		model.add(layers.Dense(units = 84))
		model.add(layers.Dense(units = 10))
		return model