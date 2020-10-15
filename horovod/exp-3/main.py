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
	data 	= np.random.rand(*data_shape)
	labels 	= np.random.rand(*label_shape)

	def _fnct(rows):
		"""
		rows is ignored as we do not want the dynamic generation of data to cause bottlenecks in measurements
		"""
		return (data, labels)
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
	    max_queue_size 	 = 3,
	    workers			 = 3, 
	    use_multiprocessing = True, ​
	)
	return

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Experiment 3 - Max throughput on scaling machine unit')
	parser.add_argument('training_rows', metavar='tr', type=int, nargs='?', help='Batch size on each generation')
	parser.add_argument('sla', metavar='sla', type=int, nargs='?', help='SLA timing in seconds')
	parser.add_argument('executors', metavar='e', type=int, nargs='?', help='Number of executors used for distributed Spark training')
	parser.add_argument('epochs', metavar='ep', type=int, nargs='?', help='Training epochs')
	parase.add_argument('validation_ratio', metavar='vr', type=int, nargs='?', help='Ratio of training data used for validation')
	args = parser.parse_args()

	TRAINING_ROWS 		= args.training_rows
	VALIDATION_RATIO 	= args.training_rows.validation_ratio
	VALIDATION_ROWS 	= int(VALIDATION_RATIO * TRAINING_ROWS)
	EXECUTORS 			= args.executors
	EPOCHS 				= args.epochs
	SLA 				= args.sla

	images_training	= random_generator(data_shape = (TRAINING_ROWS,32,32,3), label_shape = (TRAINING_ROWS,10))
	images_val 		= random_generator(data_shape = (VALIDATION_ROWS,32,32,3), label_shape = (VALIDATION_ROWS,10))
	train_gen 		= construct_randomsleep_rows(row_generator_fx = images_training, min_sleep = 0, max_sleep = 0, rows = 1)
	val_gen 		= construct_randomsleep_rows(row_generator_fx = images_val, min_sleep = 0, max_sleep = 0, rows = 1)

	hr = HorovodRunner(np = EXECUTORS)

	start 	= time.time()
	hr.run(train, train_data_generator = train_gen, val_data_generator = val_gen, epochs = EPOCHS)
	end 	= time.time()
	print(	"""
			Execution time total: {0}s, 
			SLA: {1}s, 
			Batch size: {2},
			SLA violation: {3},
			""".format(	end - start, 
						(end - start) / EPOCHS, 
						TRAINING_ROWS * EXECUTORS),
						SLA >= (end - start) / EPOCHS
					)