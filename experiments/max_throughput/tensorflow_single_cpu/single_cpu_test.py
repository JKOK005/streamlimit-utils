from TensorflowSingleCPU import TensorflowSingleCPU
from Res152SingleCPU import Res152SingleCPU
import logging

# logging.getLogger().setLevel(logging.INFO)
train = Res152SingleCPU
# train = TensorflowSingleCPU
params = { 
    "units" : 32, 
    "training_rows" : 115, 
    "training_steps_per_epoch" : 1, 
    "val_rows" : 13, 
    "val_steps_per_epoch" : 1,
    "epochs" : 50, 
    "gen_workers" : 1
    }
train.main(**params)
print(train.get_avg_epoch_timing(**params))
print(train.get_images_per_epoch(**params))
