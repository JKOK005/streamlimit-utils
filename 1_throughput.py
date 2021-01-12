from Runner import run
import csv

slas = [1, 5, 10, 20]
cores = [1, 2, 4, 8, 16, 32, 64]
batch_start = [2200, 2500, 4000, 6500, 9500, 16000, 18000]
RATIO = 0.1
EPOCH = 100

results_file = open("results", "w")
write = csv.writer(results_file)

for i in range(len(cores[1:])+1):
    for sla in slas:
        core = cores[i]
        print(core, sla)
        batch_size_for_one_sec = batch_start[i]
        current_batch = batch_size_for_one_sec * sla
        increment = current_batch / 20

        violation_counter = 0
        next_exp = False

        while not next_exp:
            _1, _2, per_epoch_time = run(current_batch, RATIO, EPOCH, sla, core)
            result = [core, sla, current_batch, per_epoch_time]
            write.writerow(result)
            print(result)
            if per_epoch_time > sla:
                violation_counter += 1
            else:
                violation_counter = 0
            
            if violation_counter > 1:
                next_exp = True
            current_batch += increment
