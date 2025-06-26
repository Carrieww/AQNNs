import time
from pathlib import Path

import numpy as np

from config import parse_args
from data_processor import DataProcessor
from experiment_runner import run_experiment
from util import output_results, verbose_print


def main():
    """
    Main function for running SPRinT experiments.
    """
    # Parse arguments and initialize
    args = parse_args()
    args.start_time = time.time()

    # Initialize data processor
    data_processor = DataProcessor(args)

    # Load and preprocess data
    Proxy_emb, Oracle_emb = data_processor.load_and_preprocess_data()

    # Prepare output path
    Path(f"./results/{args.algo}/").mkdir(parents=True, exist_ok=True)

    # Optional: Load predefined query indices for specific experiments
    query_index_list = None

    # Run experiments for multiple seeds
    for seed in range(1, 11):
        args.optimal_cost = None
        np.random.seed(seed)

        verbose_print(
            args, f"*********************** start seed {seed} ***********************"
        )

        # Prepare query data
        Oracle_dist, Proxy_dist = data_processor.prepare_query_data(
            Oracle_emb, Proxy_emb, seed, query_index_list
        )

        # Run experiment
        results = run_experiment(args, Oracle_dist, Proxy_dist, seed)

        # Output results
        output_results(args, seed, *results)


if __name__ == "__main__":
    main()
