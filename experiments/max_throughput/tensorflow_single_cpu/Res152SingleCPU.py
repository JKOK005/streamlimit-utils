import os, sys
# sys.path.append('/Users/tywan/Documents/dev/streamlimit-utils')

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
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)
        tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        tf.config.device_count = {'CPU': num_threads}

        train_imgs = ArrGenerator(img_size=np.array([training_rows, 224, 224, 3]), gen_cls=RandomArrCreator)
        train_labels = ArrGenerator(img_size=np.array([training_rows, 10]), gen_cls=RandomArrCreator)
        train_gen = DataGenerator.generate(img_gen=train_imgs, label_gen=train_labels)

        val_imgs = ArrGenerator(img_size=np.array([val_rows, 224, 224, 3]), gen_cls=RandomArrCreator)
        val_labels = ArrGenerator(img_size=np.array([val_rows, 10]), gen_cls=RandomArrCreator)
        val_gen = DataGenerator.generate(img_gen=val_imgs, label_gen=val_labels)
        
        model = Resnet.resnet152()
        opt = keras.optimizers.Adadelta()
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

        model.fit(
            x = train_gen, 
            steps_per_epoch = training_steps_per_epoch,
            epochs = epochs,
            validation_data = val_gen,
            validation_steps = val_steps_per_epoch,
            max_queue_size = 20,
            # workers = gen_workers,
            # use_multiprocessing = True,
            callbacks = [cls.time_callback]
        )

    @classmethod
    def get_images_per_epoch(cls, **kwargs):
        return (kwargs["training_rows"] * kwargs["training_steps_per_epoch"]) + (kwargs["val_rows"] * kwargs["val_steps_per_epoch"])
    
    @classmethod
    def get_avg_epoch_timing(cls, **kwargs):
        epoch_timings = cls.time_callback.get_epoch_times()
        cls.logger.info("Epoch timings {0}".format(epoch_timings))
        return np.average(epoch_timings[1:])

    @classmethod
    def main(cls, units, **kwargs):
        cls.train(num_threads=units, **kwargs)
        cls.clear_all()