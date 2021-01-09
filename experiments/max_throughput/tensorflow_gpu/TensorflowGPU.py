from models.tensorflow.Lenet5 import Lenet5
from stream_utils.ImageGenerator import *
from stream_utils.TimedCallback import TimedCallback
from tensorflow import keras
import gc
import logging
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K

class TensorflowGPU(object):
	time_callback = TimedCallback()
	logger = logging.getLogger()

	@classmethod
	def clear_all(cls):
		""" Release unused memory resources. Force garbage collection """
		K.clear_session()
		K.get_session().close()
		tf.compat.v1.reset_default_graph()
		gc.collect()
		K.set_session(tf.compat.v1.Session())
		gc.collect()

	@classmethod
	def train(cls, num_gpus, training_rows, val_rows, epochs, gen_workers):
		devices = tf.config.experimental.list_physical_devices('GPU')
		devices_names = [d.name.split("e:")[1] for d in devices]

		train_imgs   = ArrGenerator(img_size = np.array([training_rows, 32, 32, 3]), gen_cls = RandomArrCreator)
		train_labels = ArrGenerator(img_size = np.array([training_rows, 10]), gen_cls = RandomArrCreator)
		train_gen    = DataGenerator.generate(img_gen = train_imgs, label_gen = train_labels)

		val_imgs     = ArrGenerator(img_size = np.array([val_rows, 32, 32, 3]), gen_cls = RandomArrCreator)
		val_labels   = ArrGenerator(img_size = np.array([val_rows, 10]), gen_cls = RandomArrCreator)
		val_gen      = DataGenerator.generate(img_gen = val_imgs, label_gen = val_labels)

		strategy 	 = tf.distribute.MirroredStrategy(devices=devices_names[:num_gpus])

		with strategy.scope():
			model = Lenet5.get_model()
			opt = keras.optimizers.Adadelta()
			model.compile(optimizer = opt, loss = "mean_squared_error", metrics = ['accuracy'])

		model.fit(
			x 				 = train_gen,
			steps_per_epoch  = 1,
			epochs 		     = epochs,
			validation_data  = val_gen,
			validation_steps = 1,
			max_queue_size   = 20,
			workers		     = gen_workers, 
			use_multiprocessing = True,
			callbacks 		 = [cls.time_callback]
		)

	@classmethod
	def get_images_per_epoch(cls, **kwargs):
		return kwargs["training_rows"] + kwargs["val_rows"]

	@classmethod
	def get_avg_epoch_timing(cls, **kwargs):
		epoch_timings = cls.time_callback.get_epoch_times()
		cls.logger.info("Epoch timings {0}".format(epoch_timings))
		return np.average(epoch_timings)

	@classmethod
	def main(cls, units, **kwargs):
		cls.train(num_gpus = units, **kwargs)
		cls.clear_all()