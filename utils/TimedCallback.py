import tensorflow as tf
import numpy as np

class TimedCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.epoch_running_times = []
        self.epoch_start_times = {}

    def get_avg_epoch_running_time(self):
        return np.average(self.epoch_running_times)
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_times[epoch] = tf.timestamp()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_running_time = tf.timestamp() - self.epoch_start_times[epoch]
        self.epoch_running_times.append(epoch_running_time)