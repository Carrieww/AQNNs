import time
import argparse
import numpy as np
from pathlib import Path

from aquapro_util import (
    prepare_distances,
    agg_value,
    get_data,
    load_data,
    verbose_print,
)
from HT_test import HT_acc_t_test, one_proportion_z_test, HT_acc_z_test


def parse_args():
    parser = argparse.ArgumentParser(description="Hypothesis Testing Optimization")
    parser.add_argument("--Fname", type=str, default="icd9_eICU", help="Dataset name")
    parser.add_argument("--Dist_t", type=float, default=0.5, help="Distance threshold")
    parser.add_argument(
        "--Prob", type=float, default=0.95, help="Probability threshold."
    )
    parser.add_argument(
        "--PQA",
        type=str,
        default="PQA",
        choices=["PQA", "PQE"],
        help="Preprocessing quality assumption",
    )
    parser.add_argument(
        "--hypothesis_type",
        type=str,
        default="P-NNH",
        choices=["NNH", "P-NNH"],
        help="Type of hypothesis testing",
    )
    parser.add_argument(
        "--fac_list", type=str, default="0.5,1.51,0.05", help="Factor list in range."
    )
    parser.add_argument("--num_query", type=int, default=1, help="Number of queries")
    parser.add_argument("--num_sample", type=int, default=30, help="Number of samples")
    parser.add_argument("--verbose", type=bool, default=False, help="Allow print.")
    parser.add_argument("--save", type=bool, default=True, help="Allow autosave.")
    parser.add_argument(
        "--seed_cost",
        type=int,
        nargs="+",
        default=[1, 100, 2, 100, 3, 100],
        help="Seed and cost pairs",
    )
    return parser.parse_args()


def process_sample(
    args,
    seed,
    Oracle_dist,
    true_ans_D,
    sample_size,
    H1_op,
    D_attr=None,
):
    results = []
    acc_l, agg_S_l, NN_S_l, CI_S_l = [], [], [], []

    for sample_ind in range(args.num_sample):
        np.random.seed(seed * sample_ind)
        indices = np.random.choice(Oracle_dist.shape[0], sample_size, replace=False)
        oracle_dist_S = Oracle_dist[indices]
        true_ans_S = np.where(oracle_dist_S <= args.Dist_t)[0]
        S_size = len(indices)

        if args.hypothesis_type == "NNH":
            S_attr = [D_attr[i] for i in indices]
            l_S, agg_S = agg_value(S_attr, true_ans_S, args.attr_id, args.agg)
            verbose_print(args, f"Aggregation at {sample_ind}-th sample is: {agg_S}")
        elif args.hypothesis_type == "P-NNH":
            prop_S = round(true_ans_S.shape[0] / S_size, 4)
            verbose_print(args, f"prop_S at {sample_ind}-th sample is: {prop_S}")

        NN_S = len(true_ans_S)

        for fac in args.fac_list:
            if args.hypothesis_type == "NNH":
                c_time_GT = args.agg_D * fac
                verbose_print(args, f">>> c is {c_time_GT}")

                _, GT, GT_CI_l, GT_CI_h = HT_acc_t_test(
                    args, args.l_D, c_time_GT, H1_op, is_D=True
                )
                verbose_print(args, f"the ground truth to reject H0 result is : {GT}")
                align_S, _, CI_l_S, CI_h_S = HT_acc_t_test(
                    args, l_S, c_time_GT, H1_op, GT=GT, is_D=False
                )
                acc_l.append(align_S)
                agg_S_l.append(agg_S)
                CI_S_l.append(abs(GT_CI_h - GT_CI_l))
                NN_S_l.append(NN_S)
            elif args.hypothesis_type == "P-NNH":
                c_time_GT = (len(true_ans_D) / Oracle_dist.shape[0]) * fac
                verbose_print(args, f">>> c is {c_time_GT}")
                _, _, GT = one_proportion_z_test(
                    len(true_ans_D),
                    Oracle_dist.shape[0],
                    c_time_GT,
                    0.05,
                    H1_op,
                )
                verbose_print(args, f"the ground truth to reject H0 result is : {GT}")

                align_S, _ = HT_acc_z_test(
                    args,
                    "RNS",
                    true_ans_S,
                    S_size,
                    GT,
                    c_time_GT,
                    H1_op,
                )
                acc_l.append(align_S)
                agg_S_l.append(prop_S)
                CI_S_l.append(np.nan)
                NN_S_l.append(NN_S)

    results.append(
        [
            seed,
            sample_size,
            round(np.nanmean(NN_S_l), 4),
            round(np.nanmean(agg_S_l), 4),
            round(np.nanmean(CI_S_l), 4),
            round(np.nanmean(acc_l), 4),
        ]
    )
    return results


def main():
    # Parse arguments and initialize
    args = parse_args()
    start_time = time.time()

    # Load data
    Proxy_emb, Oracle_emb = load_data(args)
    args.seed_cost_dict = dict(zip(args.seed_cost[::2], args.seed_cost[1::2]))
    args.fac_list = np.arange(
        float(args.fac_list.split(",")[0]),
        float(args.fac_list.split(",")[1]),
        float(args.fac_list.split(",")[2]),
    )

    verbose_print(args, f"Prob: {args.Prob}; r: {args.Dist_t}")

    # Handle hypothesis-specific setup
    if args.hypothesis_type == "NNH":
        D_attr = get_data(filename=f"data/medical/{args.Fname}/{args.Fname}.testfull")
        args.agg = "mean"
        args.attr = "age"
        args.attr_id = 1
        verbose_print(args, f"H1: {args.agg} {args.attr} of NNs of q is tested.")
    elif args.hypothesis_type == "P-NNH":
        args.attr = "proportion"

    # Prepare output path
    save_path = f"RS-results/RS/{args.hypothesis_type}-{args.Fname}.txt"
    Path(f"./RS-results/RS/").mkdir(parents=True, exist_ok=True)

    for H1_op in ["greater", "less"]:
        if args.save:
            with open(
                save_path,
                "a",
            ) as file:
                file.write(
                    f">>>>> Attribute: {args.attr}; H1_op: {H1_op}; Dist_t: {args.Dist_t}; {args.PQA} \n"
                )
                file.write("seed\tsample size\tNN\tavg agg S\tavg CI\tavg acc\n")

        for seed, cost in args.seed_cost_dict.items():
            np.random.seed(seed)
            query_indices = np.random.choice(
                len(Oracle_emb), size=args.num_query, replace=False
            )
            Oracle_dist, Proxy_dist = prepare_distances(
                args, Oracle_emb, Proxy_emb, query_indices
            )

            args.true_ans_D = np.where(Oracle_dist <= args.Dist_t)[0]
            if args.hypothesis_type == "NNH":
                args.l_D, args.agg_D = agg_value(
                    D_attr, args.true_ans_D, args.attr_id, args.agg
                )
                verbose_print(args, f"Ground Truth Aggregation: {args.agg_D}")
            elif args.hypothesis_type == "P-NNH":
                args.prop_D = len(args.true_ans_D) / Oracle_dist.shape[0]
                verbose_print(args, f"Ground Truth Proportion: {args.prop_D}")

            # Process samples
            for sample_size in [cost]:
                verbose_print(args, f"sample size: {sample_size}")
                results = process_sample(
                    args,
                    seed,
                    Oracle_dist,
                    args.true_ans_D,
                    sample_size,
                    H1_op,
                    D_attr if args.hypothesis_type == "NNH" else None,
                )
                if args.save:
                    with open(
                        save_path,
                        "a",
                    ) as file:
                        results_str = "\t".join(map(str, results)) + "\n"
                        file.write(results_str)
                else:
                    results_str = "\t".join(map(str, results))
                    print(results_str)

    print("execution time is %.2fs" % (time.time() - start_time))


if __name__ == "__main__":
    main()
