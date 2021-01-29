from TensorflowSingleCPU import TensorflowSingleCPU
from Res152SingleCPU import Res152SingleCPU

#train = Res152SingleCPU
train = TensorflowSingleCPU
params = { 
    "units" : 64, 
    "training_rows" : 3686, 
    "training_steps_per_epoch" : 95, 
    "val_rows" : 410, 
    "val_steps_per_epoch" : 95, 
    "epochs" : 25, 
    "gen_workers" : 40
    }
train.main(**params)
print(train.get_avg_epoch_timing(**params))
print(train.get_images_per_epoch(**params))
