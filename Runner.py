import argparse
from TensorflowSingleCPU import TensorflowSingleCPU

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 1: Maximum Throughput")

    parser.add_argument(
        "batch_size",
        metavar="tr",
        type=int,
        nargs="?",
        help="Batch size on each generation",
    )
    parser.add_argument(
        "sla", 
        metavar="sla", 
        type=float, 
        nargs="?", 
        help="SLA timing in seconds"
    )
    parser.add_argument(
        "epochs", 
        metavar="ep", 
        type=int, 
        nargs="?", 
        help="Training epochs"
    )
    parser.add_argument(
        "validation_ratio",
        metavar="vr",
        type=float,
        nargs="?",
        help="Ratio of training data used for validation",
    )
    parser.add_argument(
        "num_threads",
        metavar="nt",
        type=int,
        nargs="?",
        help="Number of threads used for training",
    )

    args = parser.parse_args()

    # params
    VALIDATION_RATIO = args.validation_ratio
    VALIDATION_ROWS = int(VALIDATION_RATIO * args.batch_size)
    TRAINING_ROWS = args.batch_size - VALIDATION_ROWS
    EPOCHS = args.epochs
    SLA = args.sla
    NUM_THREADS = args.num_threads

    trainer = TensorflowSingleCPU()
    trainer.train(NUM_THREADS, TRAINING_ROWS, VALIDATION_ROWS, EPOCHS)

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
            SLA,
            args.batch_size,
            TRAINING_ROWS,
            per_epoch_time > SLA
        )
    )
