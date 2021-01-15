from Runner import run
import csv

slas = [1, 5, 10, 20]
cores = [1, 2, 4, 8, 16, 32, 64]
batch_start = [2200, 2200, 4000, 6500, 10000, 17500, 18000]
RATIO = 0.1
EPOCH = 100

core = int(input("core:"))
sla = int(input("sla:"))
batch_size = int(input("batch_size:"))

# results_file = open("results{0}".format(core), "w")
# write = csv.writer(results_file)

# for sla in slas:
#     batch_size_for_one_sec = batch_start[cores.index(core)]
#     current_batch = batch_size_for_one_sec * sla
#     increment = current_batch / 20

#     violation_counter = 0
#     next_exp = False

#     while not next_exp:
#         print(core, sla, current_batch)
#         _1, _2, per_epoch_time = run(current_batch, RATIO, EPOCH, sla, core)
#         result = [core, sla, current_batch, per_epoch_time]
#         write.writerow(result)
#         print(result)
#         if per_epoch_time > sla:
#             violation_counter += 1
#         else:
#             violation_counter = 0
        
#         if violation_counter > 0:
#             next_exp = True
#         current_batch += increment

current_batch = batch_size
increment = current_batch / 20

violation_counter = 0
next_exp = False

while not next_exp:
    print(core, sla, current_batch)
    _1, _2, per_epoch_time = run(current_batch, RATIO, EPOCH, sla, core)
    result = [core, sla, current_batch, per_epoch_time]
    write.writerow(result)
    print(result)
    if per_epoch_time > sla:
        violation_counter += 1
    else:
        violation_counter = 0
    
    if violation_counter > 0:
        next_exp = True
    current_batch += increment
