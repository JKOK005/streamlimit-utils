from models.tensorflow.Lenet5 import Lenet5
from utils.ImageGenerator import *
import argparse
import logging
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import time

class TensorflowGPU(object):
	@classmethod
	def train(cls, training_rows, val_rows, epochs, num_gpus):
		devices = tf.config.experimental.list_physical_devices('GPU')
		devices_names = [d.name.split(“e:”)[1] for d in devices]

		train_imgs   = ArrGenerator(img_size = np.array([training_rows, 32, 32, 3]), gen_cls = RandomArrCreator)
		train_labels = ArrGenerator(img_size = np.array([training_rows, 10]), gen_cls = RandomArrCreator)
		train_gen    = DataGenerator.generate(img_gen = training_imgs, label_gen = training_labels)

		val_imgs     = ArrGenerator(img_size = np.array([val_rows, 32, 32, 3]), gen_cls = RandomArrCreator)
		val_labels   = ArrGenerator(img_size = np.array([val_rows, 10]), gen_cls = RandomArrCreator)
		val_gen      = DataGenerator.generate(img_gen = val_imgs, label_gen = val_labels)

		strategy 	 = tf.distribute.MirroredStrategy(devices=devices_names[:num_gpus])

		with strategy.scope():
			model = Lenet5.get_model()
			opt = keras.optimizers.Adadelta()
			model.compile(optimizer = opt, loss = "mean_squared_error", metrics = ['accuracy'])

		model.fit_generator(
			generator        = train_gen,
			steps_per_epoch  = 1,
			epochs 		     = epochs,
			validation_data  = val_gen,
			validation_steps = 1,
			max_queue_size   = 3,
			workers		     = 3, 
			use_multiprocessing = True,
		)