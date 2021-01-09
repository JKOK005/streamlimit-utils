from models.tensorflow.Lenet5 import Lenet5
from sparkdl import HorovodRunner
from stream_utils.ImageGenerator import *
import numpy as np

class SparkHorovod(object):
  @classmethod
  def train(cls, training_rows, val_rows, epochs):
      """
      Trains model over training / validation data generators.
      We measure the average time taken to train & validate the model for each epoch
      """
      from tensorflow.keras import backend as K
      from tensorflow import keras
      import horovod.tensorflow.keras as hvd
  	
      hvd.init()
      model 	          = Lenet5.get_model()
      
      train_imgs        = ArrGenerator(img_size = np.array([training_rows, 32, 32, 3]), gen_cls = RandomArrCreator)
      train_labels      = ArrGenerator(img_size = np.array([training_rows, 10]), gen_cls = RandomArrCreator)
      train_gen         = DataGenerator.generate(img_gen = train_imgs, label_gen = train_labels)

      val_imgs          = ArrGenerator(img_size = np.array([val_rows, 32, 32, 3]), gen_cls = RandomArrCreator)
      val_labels        = ArrGenerator(img_size = np.array([val_rows, 10]), gen_cls = RandomArrCreator)
      val_gen           = DataGenerator.generate(img_gen = val_imgs, label_gen = val_labels)
    
      opt 	            = keras.optimizers.Adadelta()
      opt 	            = hvd.DistributedOptimizer(opt)
      
      model.compile(optimizer = opt, loss = "mean_squared_error", metrics = ['accuracy'])
      
      # For training 
      model.fit_generator(
        generator        = train_gen,
        steps_per_epoch  = 1,
        epochs 		       = epochs,
        validation_data  = val_gen,
        validation_steps = 1,
        max_queue_size   = 3,
        workers		       = 3, 
        use_multiprocessing = True,
      )
      
      hvd.shutdown()
      return

class SparkHorovodEntry():
  @classmethod
  def get_images_per_epoch(cls, **kwargs):
    return (kwargs["training_rows"] + kwargs["val_rows"]) * kwargs["units"]

  @classmethod
  def main(cls, units, **kwargs):
    hr = HorovodRunner(np = units)
    hr.run(SparkHorovod.train, **kwargs)