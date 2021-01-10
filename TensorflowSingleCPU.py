from models.tensorflow.Lenet5 import Lenet5
from tensorflow import keras
from utils.ImageGenerator import *
from utils.TimedCallback import TimedCallback
import logging
import tensorflow as tf


class TensorflowSingleCPU(object):

    logger = logging.getLogger()
    time_callback = TimedCallback()

    @classmethod
    def train(cls, num_thread, num_training, num_validation, num_epoch):
        # device setup
        tf.config.threading.set_intra_op_parallelism_threads(num_thread)
        tf.config.threading.set_inter_op_parallelism_threads(num_thread)

        # data prep
        train_imgs = ArrGenerator(img_size=np.array([num_training, 32, 32, 3]), gen_cls=RandomArrCreator)
        train_labels = ArrGenerator(img_size=np.array([num_training, 10]), gen_cls=RandomArrCreator)
        train_gen = DataGenerator.generate(img_gen=train_imgs, label_gen=train_labels)

        val_imgs = ArrGenerator(img_size=np.array([num_validation, 32, 32, 3]), gen_cls=RandomArrCreator)
        val_labels = ArrGenerator(img_size=np.array([num_validation, 10]), gen_cls=RandomArrCreator)
        val_gen = DataGenerator.generate(img_gen=val_imgs, label_gen=val_labels)

        # define model
        model = Lenet5.get_model()
        opt = keras.optimizers.Adadelta()
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

        # training
        model.fit(
            x=train_gen,
            steps_per_epoch=1,
            epochs=num_epoch,
            validation_data=val_gen,
            validation_steps=1,
            callbacks=[cls.time_callback]
        )

    @classmethod
    def get_avg_epoch_timing(cls):
        return cls.time_callback.get_avg_epoch_running_time()
