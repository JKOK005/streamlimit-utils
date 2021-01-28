import os, sys
sys.path.append('/Users/tywan/Documents/dev/streamlimit-utils')

from models.tensorflow.Resnets import Resnet
from stream_utils.ImageGenerator import *
from stream_utils.TimedCallback import TimedCallback
from tensorflow import keras
import gc
import logging
import numpy as np
import tensorflow as tf

class Res152SingleCPU(object):
    time_callback = TimedCallback()
    logger = logging.getLogger()

    @classmethod
    def clear_all(cls):
        tf.compat.v1.reset_default_graph()
        gc.collect()
        gc.collect()
    
    @classmethod
    def train(cls, num_threads, training_rows, training_steps_per_epoch, val_rows, val_steps_per_epoch, epochs, gen_workers):
        tf.config.threading.get_inter_op_parallelism_threads(num_threads)
        tf.config.threading.get_intra_op_parallelism_threads(num_threads)
        tf.config.device_count = {'CPU': num_threads}

        train_imgs = ArrGenerator(img_size=np.array([training_rows, 224, 224, 3]), gen_cls=RandomArrCreator)
        train_labels = ArrGenerator(img_size=np.array([training_rows, 10]), gen_cls=RandomArrCreator)
        train_gen = DataGenerator.generate(img_gen=train_imgs, label_gen=train_labels)

        val_imgs = ArrGenerator(img_size=np.array([val_rows, 224, 224, 3]), gen_cls=RandomArrCreator)
        val_labels = ArrGenerator(img_size=np.array([val_rows, 10]), gen_cls=RandomArrCreator)
        val_gen = DataGenerator.generate(img_gen=val_imgs, label_gen=val_labels)


        model = Resnet.