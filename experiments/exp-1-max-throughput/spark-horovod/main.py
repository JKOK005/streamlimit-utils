from sparkdl import HorovodRunner
from utils.ImageGenerator import *
import argparse
import logging
import numpy as np
import pandas as pd
import random
import time

logging.getLogger().setLevel(logging.INFO)

def train(training_rows, val_rows, epochs):
    """
    Trains model over training / validation data generators.
    We measure the average time taken to train & validate the model for each epoch
    """
    from tensorflow.keras import backend as K
    from tensorflow import keras
    import horovod.tensorflow.keras as hvd
	
    hvd.init()
    model 	          = Resnet().resnet50()
    
    train_imgs        = ArrGenerator(img_size = np.array([training_rows, 32, 32, 3]), gen_cls = RandomArrCreator)
    train_labels      = ArrGenerator(img_size = np.array([training_rows, 10]), gen_cls = RandomArrCreator)
    train_gen         = DataGenerator.generate(img_gen = training_imgs, label_gen = training_labels)

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

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Experiment 1: Max throughput for Spark on Horovod')
  parser.add_argument('--training_rows_min', type=int, nargs='?', default=1, help='Minimum batch size per epoch')
  parser.add_argument('--training_rows_max', type=int, nargs='?', default=8, help='Maximum batch size per epoch')
  parser.add_argument('--training_rows_inc', type=int, nargs='?', default=1, help='Batch size increment')
  parser.add_argument('--executors_min', type=int, nargs='?', default=1, help='Minimum executors')
  parser.add_argument('--executors_max', type=int, nargs='?', default=8, help='Maximum executors')
  parser.add_argument('--executors_inc', type=int, nargs='?', default=1, help='Executors increment')
  parser.add_argument('--validation_ratio', type=float, nargs='?', default=0.1, help='Ratio of training rows for validation')
  parser.add_argument('--epochs', type=int, nargs='?', default=1, help='Training epochs')
  parser.add_argument('--reps', type=int, nargs='?', default=1, help='Repetitions of experiment')
  args = parser.parse_args()
  logging.info("Training model on args: {0}".format(args))

  TRAINING_ROWS_RANGE = [i for i in range(args.training_rows_min, args.training_rows_max, args.training_rows_inc)]   # [i for i in range(35000,50000,500)]
  EXECUTORS_RANGE     = [i for i in range(args.executors_min, args.executors_max, args.executors_inc)]   # [2**i for i in range(0, -1, -1)]
  VALIDATION_RATIO    = args.validation_ratio
  EPOCHS              = args.epochs
  REPETITIONS         = args.reps
  TIMESTAMP           = int(time.time())
  RESULTS             = []

  for _ in range(REPETITIONS):
    for each_executors in EXECUTORS_RANGE:
      for each_training_rows in TRAINING_ROWS_RANGE:
        TRAINING_ROWS     = max(each_training_rows, 1)
        VALIDATION_ROWS   = max(int(VALIDATION_RATIO * TRAINING_ROWS), 1)
        EXECUTORS         = each_executors
        FAILURE_FLAG      = True

        hr = HorovodRunner(np = EXECUTORS)

        while FAILURE_FLAG:
          try:
            print("Profiling for training rows: {0}, executors: {1}".format(TRAINING_ROWS, EXECUTORS))
            start = time.time()
            hr.run(train, training_rows = TRAINING_ROWS, val_rows = VALIDATION_ROWS, epochs = EPOCHS)
            end   = time.time()
            FAILURE_FLAG = False
          except Exception as ex:
            print(ex)
            FAILURE_FLAG = True
            time.sleep(10)
          
        run_time = end - start
        per_epoch_time = (end - start) / EPOCHS
        per_epoch_imgs = TRAINING_ROWS * EXECUTORS

        RESULTS.append((EXECUTORS, run_time, per_epoch_time, per_epoch_imgs))
        print(  """
                Execution time total: {0}s, 
                SLA: {1}s, 
                Batch size: {2}
                """.format(run_time, per_epoch_time, per_epoch_imgs)
        )

        df = pd.DataFrame(RESULTS, columns = ["executors", "total_run_time", "epoch_time", "images_per_epoch"])
        df.to_csv("/dbfs/users/johan.kok/horovod/exp1/results_{0}.csv".format(TIMESTAMP), index = False)
        time.sleep(10)

  print(RESULTS)