from models.tensorflow.Lenet5 import Lenet5
from ratelimit_stream import construct_randomsleep_rows
from sparkdl import HorovodRunner
import argparse
import numpy as np
import time

def random_generator(data_shape: (int), label_shape: (int)):
	"""
	Generates random numpy arrays based on each row shape and number of rows
	"""
	data 	= np.random.rand(1, *data_shape)
	labels 	= np.random.rand(1, *label_shape)

	def _fnct(rows):
		return (np.repeat(data, repeats = rows, axis = 0), 
				np.repeat(labels, repeats = rows, axis = 0))
	return _fnct

def train(train_data_generator, val_data_generator, epochs) -> int:
	"""
	Trains model over training / validation data generators.
	We measure the average time taken to train & validate the model for each epoch
	"""
	from tensorflow.keras import backend as K
	from tensorflow import keras
	import tensorflow as tf
	import horovod.tensorflow.keras as hvd
	
	hvd.init()
	model 	= Lenet5().get_model()

	opt 	= keras.optimizers.Adadelta()
	opt 	= hvd.DistributedOptimizer(opt)

	model.compile(optimizer = opt, loss = "mean_square_error", metrics = ['accuracy'])

	model.fit_generator(
		generator        = train_data_generator(),
	    steps_per_epoch  = 1,
	    epochs 		     = epochs,
	    validation_data  = val_data_generator(),
	    validation_steps = 1,
	    max_queue_size 	 = 10,
	    workers			 = 10, 
	    use_multiprocessing = True, ​
	)
	return

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Experiment 3 - Max throughput on scaling machine unit')

	parser.add_argument('batch_size', metavar='bs', type=int, nargs='?', help='Batch size on each generation')
	parser.add_argument('sla', metavar='sla', type=int, nargs='?', help='SLA timing in seconds')
	parser.add_argument('executors', metavar='e', type=int, nargs='?', help='Number of executors used for distributed Spark training')

	MAX_ROWS 	= 100
	images_gen	= random_generator(data_shape = (32,32,3), label_shape = (10,))
	train_gen 	= construct_randomsleep_rows(row_generator_fx = images_gen, min_sleep = 0, max_sleep = 0, rows = MAX_ROWS)
	val_gen 	= construct_randomsleep_rows(row_generator_fx = images_gen, min_sleep = 0, max_sleep = 0, rows = 1)

	hr = HorovodRunner(np=15)
	hr.run(train, train_data_generator = train_gen, val_data_generator = val_gen, epochs=10)
