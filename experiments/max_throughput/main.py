from experiments.max_throughput.spark_horovod.SparkHorovod import *
from experiments.max_throughput.tensorflow_gpu.TensorflowGPU import *
import argparse
import logging
import os
import pandas as pd
import time

logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Experiment 1: Max throughput')
  parser.add_argument('--experiment_type', type=int, choices=[1], nargs='?', default=1, help='1 - TF-GPU, 2 - TF-CPU, 3 - Spark-Horovod')
  parser.add_argument('--training_rows_min', type=int, nargs='?', default=1, help='Minimum batch size per epoch')
  parser.add_argument('--training_rows_max', type=int, nargs='?', default=8, help='Maximum batch size per epoch')
  parser.add_argument('--training_rows_inc', type=int, nargs='?', default=1, help='Batch size increment')
  parser.add_argument('--machine_units_min', type=int, nargs='?', default=1, help='Minimum number of machine units')
  parser.add_argument('--machine_units_max', type=int, nargs='?', default=8, help='Maximum number of machine units')
  parser.add_argument('--machine_units_inc', type=int, nargs='?', default=1, help='Machine units increment')
  parser.add_argument('--validation_ratio', type=float, nargs='?', default=0.1, help='Ratio of training rows for validation')
  parser.add_argument('--epochs', type=int, nargs='?', default=1, help='Training epochs')
  parser.add_argument('--reps', type=int, nargs='?', default=1, help='Repetitions of experiment')
  parser.add_argument('--sleep_interval', type=int, nargs='?', default=10, help='Interval between runs')
  parser.add_argument('--out_dir', type=str, nargs='?', default='.', help='Save results to directory')
  args = parser.parse_args()
  logging.info("Training model on args: {0}".format(args))

  TRAINING_ROWS_RANGE = [i for i in range(args.training_rows_min, args.training_rows_max, args.training_rows_inc)]   # [i for i in range(35000,50000,500)]
  UNITS_RANGE         = [i for i in range(args.machine_units_min, args.machine_units_max, args.machine_units_inc)]   # [2**i for i in range(0, -1, -1)]
  EPOCHS              = args.epochs
  EXPERIMENT_TYPE     = args.experiment_type
  REPETITIONS         = args.reps
  VALIDATION_RATIO    = args.validation_ratio
  OUT_DIR             = args.out_dir
  SLEEP_INTERVAL      = args.sleep_interval

  TIMESTAMP           = int(time.time())
  RESULTS             = []

  training_cls        = None
  if EXPERIMENT_TYPE == 1:
    training_cls      = TensorflowGPU
  elif EXPERIMENT_TYPE == 3:
    # This option is disabled as Running on Databricks incurs the error: Missing master URL. 
    # Probably due to the way Spark is set up in Databricks
    # Workaround is to run the script in Databrick's notebook instead of executing this main script
    training_cls      = SparkHorovodEntry

  for _ in range(REPETITIONS):
    for each_units in UNITS_RANGE:
      for each_training_rows in TRAINING_ROWS_RANGE:
        TRAINING_ROWS     = max(each_training_rows, 1)
        VALIDATION_ROWS   = max(int(VALIDATION_RATIO * TRAINING_ROWS), 1)
        FAILURE_FLAG      = True
        params            = {"units" : each_units, "training_rows" : TRAINING_ROWS, "val_rows" : VALIDATION_ROWS, "epochs" : EPOCHS}

        while FAILURE_FLAG:
          try:
            logging.info("Training instance: {0}".format(params))
            start = time.time()
            training_cls.main(**params)
            end   = time.time()
            FAILURE_FLAG = False
          except Exception as ex:
            logging.error("Failed due to {1}".format(ex))
            FAILURE_FLAG = True
            time.sleep(SLEEP_INTERVAL)
          
        run_time = end - start
        per_epoch_time = (end - start) / EPOCHS
        per_epoch_imgs = training_cls.get_images_per_epoch(**params)

        RESULTS.append((each_units, run_time, per_epoch_time, per_epoch_imgs))
        logging.info("""
                Execution time total: {0}s, 
                SLA: {1}s, 
                Batch size: {2}
          """.format(run_time, per_epoch_time, per_epoch_imgs)
        )

        df = pd.DataFrame(RESULTS, columns = ["machine_units", "total_run_time", "epoch_time", "images_per_epoch"])
        df.to_csv(os.path.join(OUT_DIR, "results_{0}.csv".format(TIMESTAMP)), index = False)
        time.sleep(SLEEP_INTERVAL)

  logging.info(RESULTS)