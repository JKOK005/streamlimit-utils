from models.tensorflow.Lenet5 import Lenet5
from ratelimit_stream import construct_randomsleep_rows
import argparse
import numpy as np
import time

def random_generator(shape: (int)):
	"""
	Generates random numpy arrays based on each row shape and number of rows
	"""
	def _fnct(rows):
		return np.random.rand(rows, *shape)
	return _fnct

def train(train_data_generator, val_data_generator, epoch) -> int:
	"""
	Trains model over training / validation data generators.
	We measure the average time taken to train & validate the model for each epoch
	"""
	pass

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Experiment 3 - Max throughput on scaling machine unit')

	parser.add_argument('batch_size', metavar='bs', type=int, nargs='?', help='Batch size on each generation')
	parser.add_argument('sla', metavar='sla', type=int, nargs='?', help='SLA timing in seconds')

