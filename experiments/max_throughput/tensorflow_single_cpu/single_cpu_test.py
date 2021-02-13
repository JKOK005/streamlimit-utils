from TensorflowSingleCPU import TensorflowSingleCPU
from Res152SingleCPU import Res152SingleCPU
import logging

logging.getLogger().setLevel(logging.INFO)

train = Res152SingleCPU
# train = TensorflowSingleCPU
params = { 
    "units" : 64, 
    "training_rows" : 9, 
    "training_steps_per_epoch" : 40, 
    "val_rows" : 1, 
    "val_steps_per_epoch" : 40, 
    "epochs" : 40, 
    "gen_workers" : 2
    }
train.main(**params)
print(train.get_avg_epoch_timing(**params))
print(train.get_images_per_epoch(**params))
