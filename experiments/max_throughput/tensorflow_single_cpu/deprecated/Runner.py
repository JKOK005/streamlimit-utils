import argparse
from TensorflowSingleCPU import TensorflowSingleCPU

def run(batch_size, validation_ratio, epochs, sla, threads):
    
    validation_rows = int(batch_size * validation_ratio)
    training_rows = int(batch_size - validation_rows)

    trainer = TensorflowSingleCPU()
    trainer.train(threads, training_rows, validation_rows, epochs)

    per_epoch_time = trainer.get_avg_epoch_timing()

    print(
        """
        per epoch time: {0},
        SLA: {1},
        batch size: {2},
        training rows: {3},
        SLA violated: {4}
        """.format(
            per_epoch_time,
            sla,
            batch_size,
            training_rows,
            per_epoch_time > sla
        )
    )
    return sla, batch_size, per_epoch_time