import tensorflow as tf
from datetime import datetime

class TimedCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.epoch_times        = []
        self.epoch_start_log    = {}
    
    def get_epoch_times(self):
        return self.epoch_times

    def on_epoch_begin(self, epoch, logs = {}):
        self.epoch_start_log[epoch] = tf.timestamp()

    def on_epoch_end(self, epoch, logs = {}):
        epoch_timing = tf.timestamp() - self.epoch_start_log[epoch]
        self.epoch_times.append(epoch_timing)