from TensorflowSingleCPU import TensorflowSingleCPU
from Res152SingleCPU import Res152SingleCPU
import logging

logging.getLogger().setLevel(logging.INFO)
#train = Res152SingleCPU
train = TensorflowSingleCPU
params = { 
    "units" : 32, 
    "training_rows" : 3686, 
    "training_steps_per_epoch" : 10, 
    "val_rows" : 410, 
    "val_steps_per_epoch" : 10,
    "epochs" : 40, 
    "gen_workers" : 2
    }
train.main(**params)
print(train.get_avg_epoch_timing(**params))
print(train.get_images_per_epoch(**params))
