from experiments.max_throughput.spark_horovod.SparkHorovod import *
from experiments.max_throughput.tensorflow_gpu.TensorflowGPU import *
import argparse
import logging
import math
import os
import pandas as pd
import time

logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Experiment 1: Max throughput')
  parser.add_argument('--experiment_type', type=int, choices=[1], nargs='?', default=1, help='1 - TF-GPU, 2 - TF-CPU, 3 - Spark-Horovod')
  parser.add_argument('--training_rows_min', type=int, nargs='?', default=1, help='Minimum data size per epoch')
  parser.add_argument('--training_rows_max', type=int, nargs='?', default=8, help='Maximum data size per epoch')
  parser.add_argument('--training_rows_inc', type=int, nargs='?', default=1, help='Data size increment')
  parser.add_argument('--batch_size', type=int, nargs='?', default=4096, help='Batch size for each training')
  parser.add_argument('--machine_units_min', type=int, nargs='?', default=1, help='Minimum number of machine units')
  parser.add_argument('--machine_units_max', type=int, nargs='?', default=8, help='Maximum number of machine units')
  parser.add_argument('--machine_units_inc', type=int, nargs='?', default=1, help='Machine units increment')
  parser.add_argument('--validation_ratio', type=float, nargs='?', default=0.1, help='Ratio of training rows for validation')
  parser.add_argument('--generator_workers', type=int, nargs='?', default=1, help='Workers for generating images')
  parser.add_argument('--epochs', type=int, nargs='?', default=1, help='Training epochs')
  parser.add_argument('--reps', type=int, nargs='?', default=1, help='Repetitions of experiment')
  parser.add_argument('--sleep_interval', type=int, nargs='?', default=10, help='Interval between runs')
  parser.add_argument('--out_dir', type=str, nargs='?', default='.', help='Save results to directory')
  args = parser.parse_args()
  logging.info("Training model on args: {0}".format(args))

  TRAINING_ROWS_RANGE = [i for i in range(args.training_rows_min, args.training_rows_max, args.training_rows_inc)]   # [i for i in range(35000,50000,500)]
  UNITS_RANGE         = [i for i in range(args.machine_units_min, args.machine_units_max, args.machine_units_inc)]   # [2**i for i in range(0, -1, -1)]
  BATCH_SIZE          = args.batch_size
  EPOCHS              = args.epochs
  EXPERIMENT_TYPE     = args.experiment_type
  REPETITIONS         = args.reps
  VALIDATION_RATIO    = args.validation_ratio
  OUT_DIR             = args.out_dir
  SLEEP_INTERVAL      = args.sleep_interval
  GEN_WORKERS         = args.generator_workers

  TIMESTAMP           = int(time.time())
  RESULTS             = []

  training_cls        = None
  if EXPERIMENT_TYPE == 1:
    training_cls      = TensorflowGPU
  elif EXPERIMENT_TYPE == 3:
    # This option is disabled as Running on Databricks incurs the error: Missing master URL. 
    # Probably due to the way Spark is set up in Databricks
    # Workaround is to run the script in Databrick's notebook instead of executing this main script
    # training_cls      = SparkHorovodEntry
    training_cls      = None

  for _ in range(REPETITIONS):
    for each_units in UNITS_RANGE:
      for each_training_rows in TRAINING_ROWS_RANGE:
        TRAINING_ROWS     = max(int((1 -VALIDATION_RATIO) * each_training_rows), 1)
        VALIDATION_ROWS   = max(int(VALIDATION_RATIO * each_training_rows), 1)
        training_steps_per_epoch    = math.ceil(TRAINING_ROWS / BATCH_SIZE)
        val_steps_per_epoch         = math.ceil(VALIDATION_ROWS / BATCH_SIZE)
        FAILURE_FLAG      = True
        params            = { "units" : each_units, "training_rows" : BATCH_SIZE, "training_steps_per_epoch" : training_steps_per_epoch, 
                              "val_rows" : VALIDATION_ROWS, "training_steps_per_epoch" : val_steps_per_epoch, "epochs" : EPOCHS, 
                              "gen_workers" : GEN_WORKERS}

        while FAILURE_FLAG:
          try:
            logging.info("Training instance: {0}".format(params))
            training_cls.main(**params)
            FAILURE_FLAG = False
          except Exception as ex:
            logging.error("Failed due to {0}".format(ex))
            FAILURE_FLAG = False
            time.sleep(SLEEP_INTERVAL)
          
        per_epoch_time = training_cls.get_avg_epoch_timing()
        per_epoch_imgs = training_cls.get_images_per_epoch(**params)

        RESULTS.append((each_units, per_epoch_time, per_epoch_imgs))
        logging.info("""
                SLA: {0}s, 
                Batch size: {1}
          """.format(per_epoch_time, per_epoch_imgs)
        )

        df = pd.DataFrame(RESULTS, columns = ["machine_units", "per_epoch_time", "images_per_epoch"])
        df.to_csv(os.path.join(OUT_DIR, "results_{0}.csv".format(TIMESTAMP)), index = False)
        time.sleep(SLEEP_INTERVAL)

  logging.info(RESULTS)