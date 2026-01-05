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

    # Run experiments for 10 seeds from 1 to 10
    # we use seed = 2024 for scalability experiments
    brute_force_time_l = []
    for seed in range(1, 11): 
        args.optimal_cost = None
        np.random.seed(seed)

        verbose_print(
            args, f"*********************** start seed {seed} ***********************"
        )

        # Prepare query data
        Oracle_dist, Proxy_dist, brute_force_time = data_processor.prepare_query_data(
            Oracle_emb, Proxy_emb, seed, query_index_list
        )
        brute_force_time_l.append(brute_force_time)
        verbose_print(args, f"Brute force time: {brute_force_time:.4f} seconds")

        # Run experiment
        batch_results = run_experiment(args, Oracle_dist, Proxy_dist, seed)

        # output results for each aggregation type
        for agg_type, result_tuple in batch_results.items():
            args.agg = agg_type
            
            (
                avg_execution_time,
                avg_error,
                avg_absError,
                avg_acc,
                avg_rec,
                avg_prec,
                avg_fix_rec,
                avg_fix_prec,
                avg_NN_algo,
                avg_agg,
                var_agg,
                avg_NN_S,
                avg_agg_S,
                prec_rec_diff,
                avg_CI,
                avg_f1,
                avg_fix_f1,
                cannot_times,
            ) = result_tuple

            # output results
            output_results(
                args,
                seed,
                avg_execution_time,
                avg_error,
                avg_absError,
                avg_NN_algo,
                avg_agg,
                var_agg,
                avg_NN_S,
                avg_agg_S,
                prec_rec_diff,
                avg_CI,
                avg_f1,
                avg_fix_f1,
                avg_acc,
                avg_rec,
                avg_prec,
                avg_fix_rec,
                avg_fix_prec,
                cannot_times,
            )

    verbose_print(args, f"Average brute force time: {np.mean(brute_force_time_l):.4f} seconds")


if __name__ == "__main__":
    main()
