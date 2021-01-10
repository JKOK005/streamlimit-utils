from Runner import run

slas = [1, 5, 10, 20]
cores = [1, 2, 4, 8, 16, 32, 64]
batch_range = []
RATIO = 0.1
EPOCH = 100

# for sla in slas:
#     for core in cores:
#         run(2000, RATIO, EPOCH, sla, core)
run(2000, RATIO, EPOCH, 1, 1)