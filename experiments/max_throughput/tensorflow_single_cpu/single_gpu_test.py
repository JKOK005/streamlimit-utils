from TensorflowSingleCPU import TensorflowSingleCPU
from Res152SingleCPU import Res152SingleCPU

train = Res152SingleCPU
params = { 
    "units" : 6, 
    "training_rows" : 900, "training_steps_per_epoch" : 2, 
    "val_rows" : 100, "val_steps_per_epoch" : 1, 
    "epochs" : 15, "gen_workers" : 1}
train.main(**params)
print(train.get_avg_epoch_timing(**params))
print(train.get_images_per_epoch(**params))