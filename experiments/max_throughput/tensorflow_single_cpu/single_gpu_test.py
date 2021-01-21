from TensorflowSingleCPU import TensorflowSingleCPU

train = TensorflowSingleCPU
params = { 
    "units" : 6, 
    "training_rows" : 9000, "training_steps_per_epoch" : 2, 
    "val_rows" : 1000, "val_steps_per_epoch" : 1, 
    "epochs" : 15, "gen_workers" : 1}
train.main(**params)
print(train.get_avg_epoch_timing(**params))
print(train.get_images_per_epoch(**params))