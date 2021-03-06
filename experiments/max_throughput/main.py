import argparse
import logging
import math
import os
import pandas as pd
import sys
import time

logging.getLogger().setLevel(logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Experiment 1: Max throughput')
  parser.add_argument('--experiment_type', type=int, choices=[1, 2], nargs='?', default=1, help='1 - TF-GPU, 2 - TF-CPU, 3 - Spark-Horovod')
  parser.add_argument('--rows_min', type=int, nargs='?', default=1, help='Minimum data size per epoch')
  parser.add_argument('--rows_max', type=int, nargs='?', default=8, help='Maximum data size per epoch')
  parser.add_argument('--rows_inc', type=int, nargs='?', default=1, help='Data size increment')
  parser.add_argument('--steps_per_epoch_min', type=int, nargs='?', default=1, help='Minimum steps per epoch')
  parser.add_argument('--steps_per_epoch_max', type=int, nargs='?', default=8, help='Maximum steps per epoch')
  parser.add_argument('--steps_per_epoch_inc', type=int, nargs='?', default=1, help='Steps per epoch increment')
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

  ROWS_RANGE            = [i for i in range(args.rows_min, args.rows_max, args.rows_inc)]   # [i for i in range(35000,50000,500)]
  UNITS_RANGE           = [i for i in range(args.machine_units_min, args.machine_units_max, args.machine_units_inc)]   # [2**i for i in range(0, -1, -1)]
  STEPS_PER_EPOCH_RANGE = [i for i in range(args.steps_per_epoch_min, args.steps_per_epoch_max, args.steps_per_epoch_inc)]
  EPOCHS                = args.epochs
  EXPERIMENT_TYPE       = args.experiment_type
  REPETITIONS           = args.reps
  VALIDATION_RATIO      = args.validation_ratio
  OUT_DIR               = args.out_dir
  SLEEP_INTERVAL        = args.sleep_interval
  GEN_WORKERS           = args.generator_workers

  TIMESTAMP           = int(time.time())
  RESULTS             = []

  training_cls        = None
  if EXPERIMENT_TYPE == 1:
    from experiments.max_throughput.tensorflow_gpu.TensorflowGPU import *
    training_cls      = TensorflowGPU

  elif EXPERIMENT_TYPE == 2:
    from experiments.max_throughput.tensorflow_single_cpu.TensorflowSingleCPU import *
    training_cls      = TensorflowSingleCPU

  elif EXPERIMENT_TYPE == 3:
    # This option is disabled as Running on Databricks incurs the error: Missing master URL. 
    # Probably due to the way Spark is set up in Databricks
    # Workaround is to run the script in Databrick's notebook instead of executing this main script
    # training_cls      = SparkHorovodEntry
    from experiments.max_throughput.spark_horovod.SparkHorovod import *
    training_cls      = None

  for _ in range(REPETITIONS):
    for each_units in UNITS_RANGE:
      for each_rows in ROWS_RANGE:
        for each_steps_per_epoch in STEPS_PER_EPOCH_RANGE:
          TRAINING_ROWS               = max(int((1 -VALIDATION_RATIO) * each_rows), 1)
          VALIDATION_ROWS             = max(int(VALIDATION_RATIO * each_rows), 1)
          FAILURE_FLAG      = True
          params            = { "units" : each_units, "training_rows" : TRAINING_ROWS, "training_steps_per_epoch" : each_steps_per_epoch, 
                                "val_rows" : VALIDATION_ROWS, "val_steps_per_epoch" : each_steps_per_epoch, "epochs" : EPOCHS, 
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