import time
import numpy as np
from numba import njit
from pathlib import Path
from math import ceil
from config import parse_args
import copy
import pandas as pd
from hyper_parameter import norm_scale, std_offset
from util import (
    prepare_distances,
    agg_value,
    get_data,
    load_data,
    preprocess_topk_phi,
    verbose_print,
    array_union,
    set_diff,
    output_results,
    compute_f1_score,
)
from hypothesis_testing import HT_acc_t_test, one_proportion_z_test, HT_acc_z_test
from baselines import (
    test_topk,
    test_PQE,
    SUPG,
)


@njit
def SPRinT(Dist_t, Prob, fix_sample, oracle_dist, phi, topk, pt=0.9, pilots=None):
    true_ans = np.where(oracle_dist <= Dist_t)[0]
    if len(true_ans) == 0:
        return 0, 0, 0, np.empty(0, dtype=np.int64), 0, 0

    pbs = np.zeros(len(phi) + 1)
    k_star = 0

    for i in range(1, len(phi) + 1):
        if i == 1:
            pbs[0] = 1 - phi[0]
            pbs[1] = phi[0]
        else:
            shift_pbs = np.roll(pbs, 1) * phi[i - 1]
            pbs = pbs * (1 - phi[i - 1]) + shift_pbs

        idx_s = ceil(i * pt)
        precis_prob = np.sum(pbs[idx_s : i + 1])

        if precis_prob >= Prob:
            k_star = i

    if k_star == 0:
        return 1, 0, k_star, topk[:k_star], 0, 0

    ans = topk[:k_star]

    # fixed sample's precision
    fix_true_ans = fix_sample[np.where(oracle_dist[fix_sample] <= Dist_t)[0]]
    fix_true_pos = len(np.intersect1d(ans, fix_true_ans))

    if len(np.intersect1d(ans, fix_sample)) == 0:
        fix_precision = 0
    else:
        fix_precision = fix_true_pos / len(np.intersect1d(ans, fix_sample))
    if len(fix_true_ans) == 0:
        fix_recall = 0
    else:
        fix_recall = fix_true_pos / len(fix_true_ans)

    if pilots is None:
        ans = topk[:k_star]
    else:
        pilots_false = pilots[np.where(oracle_dist[pilots] > Dist_t)[0]]
        ans = set_diff(array_union(topk[:k_star], pilots), pilots_false)

    true_pos = len(np.intersect1d(ans, true_ans))
    if len(ans) == 0:
        precision = 0
    else:
        precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, k_star, ans, fix_precision, fix_recall


def test_SPRinT(args, oracle_dist, proxy_dist, pt, find_rt=True):
    if find_rt:
        args.samples = None
    else:
        args.samples = args.fix_sample

    est_scale = (
        np.std(oracle_dist[args.fix_sample] - proxy_dist[args.fix_sample]) + std_offset
    )
    topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=est_scale, t=args.Dist_t)

    precision, recall, _, ans, fix_prec, fix_rec = SPRinT(
        args.Dist_t,
        args.Prob,
        args.fix_sample,
        oracle_dist,
        phi,
        topk,
        pt=pt,
        pilots=args.samples,
    )

    verbose_print(
        args,
        f"true precision is {precision}, true recall is {recall}, and the precision for fixed sample is {fix_prec}, and recall for fixed sample is {fix_rec}",
    )

    return precision, recall, _, ans, fix_prec, fix_rec


def run_experiment(
    args,
    Oracle_dist,
    Proxy_dist,
    seed,
):
    acc_l = []
    relativeError_l = []
    absoluteError_l = []
    recall_l = []
    precision_l = []
    fix_prec_l = []
    fix_rec_l = []
    agg_l = []
    agg_S_l = []
    NN_S_l = []
    NN_RT_l = []
    time_l = []
    cannot_times_l = []

    if args.agg in ["avg", "var", "sum"]:
        CI_l = []
        f1_l = []
        fix_f1_l = []

    for i in range(args.num_sample):
        start_sample = time.time()
        np.random.seed(seed * i)

        # --- Sample a subset S from the full distributions ---
        indices = np.random.choice(Oracle_dist.shape[0], args.s, replace=False)
        oracle_dist_S = Oracle_dist[indices]
        proxy_dist_S = Proxy_dist[indices]
        args.true_ans_S = np.where(oracle_dist_S <= args.Dist_t)[0]
        args.NN_S = len(args.true_ans_S)
        verbose_print(
            args, f">>> algo {args.algo} | sample {i} | Find NN in S is {args.NN_S}"
        )

        # --- Compute Aggregation Value in S ---
        if args.agg == "pct":
            args.agg_S = len(args.true_ans_S) / oracle_dist_S.shape[0]
            verbose_print(args, f"the prop in S is {args.agg_S}")
        else:
            args.S_attr = [args.D_attr[i] for i in indices]
            args.l_S, args.agg_S = agg_value(
                args.S_attr, args.true_ans_S, args.attr_id, args.agg
            )
            _, args.agg_S_full = agg_value(
                args.S_attr, range(len(args.S_attr)), args.attr_id, args.agg
            )
            verbose_print(
                args,
                f"The number of NN in S is {len(args.true_ans_S)} ({(len(args.true_ans_S)/proxy_dist_S.shape[0])*100}%), the aggregation of true NN is {args.agg_S} and the aggregated value of all data in S is {args.agg_S_full}",
            )

        # --- Choose pilot samples ---
        args.fix_sample = np.random.choice(
            len(oracle_dist_S), size=int(args.s_p), replace=False
        )
        args.oracle_dist_S_p = oracle_dist_S[args.fix_sample]
        args.proxy_dist_S_p = proxy_dist_S[args.fix_sample]
        pilot_nn = len(np.where(args.oracle_dist_S_p <= args.Dist_t)[0])
        verbose_print(args, f"Number of NN in pilot sample: {pilot_nn}")

        start_query_sample = time.time()

        # --- Run the selected algorithm ---
        if args.algo == "SPRinT":
            args.optimal_cost = args.s_p
            before_rt = time.time()
            verbose_print(
                args,
                f"Data preparation time: {round(time.time() - start_sample, 2)} sec",
            )
            args.rt, CANNOT = find_optimal_rt(args, oracle_dist_S, proxy_dist_S)
            verbose_print(
                args, f"Optimal rt search time: {round(time.time() - before_rt, 2)} sec"
            )
            before_algo = time.time()
            prec, rec, _, ANS, fix_prec, fix_rec = test_SPRinT(
                args, oracle_dist_S, proxy_dist_S, args.rt, find_rt=False
            )

        elif args.algo in ["PQA-RT", "PQA-PT"]:
            args.optimal_cost = args.s_p
            before_algo = time.time()
            args.target = (
                args.recall_target if args.algo == "PQA-RT" else args.precision_target
            )
            prec, rec, _, ANS, _, _ = test_PQE(
                args, oracle_dist_S, proxy_dist_S, args.algo[-2:], args.target
            )
            fix_prec, fix_rec, CANNOT = np.nan, np.nan, np.nan

        elif args.algo in ["SUPG-RT", "SUPG-PT"]:
            args.optimal_cost = args.s_p
            before_algo = time.time()
            args.target = (
                args.recall_target if args.algo == "SUPG-RT" else args.precision_target
            )
            prec, rec, _, _, ANS = SUPG(
                oracle_dist_S,
                proxy_dist_S,
                args.Dist_t,
                args.target,
                args.Prob,
                cost=args.s_p,
                query_type=args.algo[-2:],
            )
            fix_prec, fix_rec, CANNOT = np.nan, np.nan, np.nan

        elif args.algo == "TopK":
            before_algo = time.time()
            prec, rec, args.optimal_cost, ANS = test_topk(
                oracle_dist=oracle_dist_S,
                proxy_dist=proxy_dist_S,
                scale=norm_scale,
                t=args.Dist_t,
                prob=args.Prob,
            )
            fix_prec, fix_rec, CANNOT = np.nan, np.nan, np.nan
        else:
            raise ValueError("Unknown algorithm specified in args.algo")

        args.NN = len(ANS)
        verbose_print(
            args,
            f"{args.algo} results | Recall: {rec}, Precision: {prec}, "
            f"Fix Prec: {fix_prec}, Fix Rec: {fix_rec} (Cost: {args.s_p})",
        )
        verbose_print(
            args, f"{args.algo} search time: {round(time.time() - before_algo, 2)} sec"
        )

        recall_l.append(rec)
        precision_l.append(prec)
        fix_prec_l.append(fix_prec)
        fix_rec_l.append(fix_rec)

        # --- Compute the approximated aggregation over found NN ---
        if args.agg == "pct":
            approx_agg_S = round(args.NN / proxy_dist_S.shape[0], 4)
            verbose_print(args, f"the approx prop is {approx_agg_S}")
        else:
            L_S, approx_agg_S = agg_value(args.S_attr, ANS, args.attr_id, args.agg)

            verbose_print(
                args,
                f"The number of NN by {args.algo} is {args.NN} ({(args.NN/proxy_dist_S.shape[0])*100}%), the approximated aggregated value is {approx_agg_S}",
            )

            if (prec + rec) == 0:
                f1 = 0
            else:
                f1 = compute_f1_score(args, prec, rec)
            if np.isnan(fix_prec):
                fix_f1 = np.nan
            elif (fix_prec + fix_rec) == 0:
                fix_f1 = 0
            else:
                fix_f1 = compute_f1_score(args, fix_prec, fix_rec)

            f1_l.append(f1)
            fix_f1_l.append(fix_f1)

        time_l.append(round(time.time() - start_query_sample, 2))

        # --- Hypothesis Testing ---
        before_HT = time.time()
        for fac in args.fac_list:
            if not (
                (args.hypothesis_type == "P-NNH" and args.agg == "pct")
                or (args.hypothesis_type == "NNH" and args.agg == "avg")
            ):
                print(f"no HT application for {args.hypothesis_type} and {args.agg}")
                break
            for H1_op in ["greater", "less"]:
                if args.hypothesis_type == "P-NNH" and args.agg == "pct":
                    c_time_GT = (len(args.true_ans_D) / Oracle_dist.shape[0]) * fac

                    _, _, GT = one_proportion_z_test(
                        len(args.true_ans_D),
                        Oracle_dist.shape[0],
                        c_time_GT,
                        0.05,
                        H1_op,
                    )

                    rt_align, _ = HT_acc_z_test(
                        args,
                        "PQE-RT",
                        ANS,
                        oracle_dist_S.shape[0],
                        GT,
                        c_time_GT,
                        H1_op,
                    )

                    acc_l.append(rt_align)

                elif args.hypothesis_type == "NNH" and args.agg == "avg":
                    c_time_GT = args.agg_D * fac

                    _, GT, _, _ = HT_acc_t_test(
                        args, args.l_D, c_time_GT, H1_op, is_D=True
                    )
                    rt_align, _, rt_CI_l, rt_CI_h = HT_acc_t_test(
                        args, L_S, c_time_GT, H1_op, GT=GT, is_D=False
                    )

                    acc_l.append(rt_align)
                    CI_l.append(rt_CI_h - rt_CI_l)
                    f1_l.append(f1)
                    fix_f1_l.append(fix_f1)

        verbose_print(
            args, f"time of hypothesis testing {round(time.time() - before_HT,2)}"
        )

        # --- Error Calculation ---
        if args.agg == "sum":
            approx_agg_S = approx_agg_S * (Oracle_dist.shape[0] / args.s)

        relativeError = abs(approx_agg_S - args.agg_D) / args.agg_D * 100
        relativeError_l.append(relativeError)
        absoluteError = abs(approx_agg_S - args.agg_D)
        absoluteError_l.append(absoluteError)

        agg_l.append(approx_agg_S)
        agg_S_l.append(args.agg_D)
        NN_S_l.append(args.NN_S)
        NN_RT_l.append(args.NN)
        cannot_times_l.append(CANNOT)

    # --- Compute Overall Statistics ---
    avg_acc = np.nanmean(acc_l)
    avg_absError = np.nanmean(absoluteError_l)
    avg_error = np.nanmean(relativeError_l)
    avg_rec = np.nanmean(recall_l)
    avg_prec = np.nanmean(precision_l)
    avg_fix_rec = np.nanmean(fix_rec_l)
    avg_fix_prec = np.nanmean(fix_prec_l)
    avg_NN_S = np.nanmean(NN_S_l)
    avg_NN_RT = np.nanmean(NN_RT_l)
    cannot_times = np.nanmean(cannot_times_l)
    avg_execution_time = np.nanmean(time_l[1:])

    verbose_print(
        args, f"Average relative error over {args.num_sample} runs: {avg_error}"
    )
    verbose_print(
        args, f"Average absolute error over {args.num_sample} runs: {avg_absError}"
    )
    verbose_print(args, f"Average HT accuracy over {args.num_sample} runs: {avg_acc}")
    verbose_print(args, f"Average recall: {avg_rec}")
    verbose_print(args, f"Average precision: {avg_prec}")
    verbose_print(args, f"Average fixed recall: {avg_fix_rec}")
    verbose_print(args, f"Average fixed precision: {avg_fix_prec}")
    verbose_print(args, f"Average NN in S: {avg_NN_S}")
    verbose_print(args, f"Average NN by {args.algo} in S: {avg_NN_RT}")
    verbose_print(args, f"Average execution time: {avg_execution_time}")

    avg_agg = np.nanmean(agg_l)
    var_agg = np.nanvar(agg_l)
    avg_agg_S = np.nanmean(agg_S_l)
    std_deviation = np.nanstd(relativeError_l, ddof=1)
    standard_error = std_deviation / np.sqrt(np.sum(~np.isnan(relativeError_l)))

    # --- Return Results Based on Aggregation Type ---
    if args.agg == "pct":
        verbose_print(args, f"Avg prop_S: {avg_agg} with variance {round(var_agg,4)}")
        return (
            avg_execution_time,
            avg_error,
            avg_absError,
            avg_acc,
            avg_rec,
            avg_prec,
            avg_fix_rec,
            avg_fix_prec,
            avg_NN_RT,
            avg_agg,
            var_agg,
            avg_NN_S,
            avg_agg_S,
            standard_error,
            None,
            None,
            None,
            cannot_times,
        )
    else:
        avg_CI = np.mean(CI_l)
        avg_f1 = np.mean(f1_l)
        avg_fix_f1 = np.mean(fix_f1_l)
        verbose_print(
            args, f"Avg aggregate value: {avg_agg} with variance {round(var_agg,4)}"
        )
        verbose_print(args, f"Avg aggregate in S: {avg_agg_S}")
        verbose_print(args, f"Avg CI: {avg_CI}")
        verbose_print(args, f"Avg F1 score: {avg_f1}")
        verbose_print(args, f"Avg fixed F1 score: {avg_fix_f1}")
        return (
            avg_execution_time,
            avg_error,
            avg_absError,
            avg_acc,
            avg_rec,
            avg_prec,
            avg_fix_rec,
            avg_fix_prec,
            avg_NN_RT,
            avg_agg,
            var_agg,
            avg_NN_S,
            avg_agg_S,
            standard_error,
            avg_CI,
            avg_f1,
            avg_fix_f1,
            None,
        )


def f1_score(args, t, oracle_dist_S, proxy_dist_S):
    _,_,_,_, fix_prec, fix_rec = test_SPRinT(args, oracle_dist_S, proxy_dist_S, t)
    if fix_prec + fix_rec == 0:
        return 0
    else:
        return compute_f1_score(args, fix_prec, fix_rec)


def find_optimal_rt(args, oracle_dist_S, proxy_dist_S, rt=0.0001):
    CANNOT = 0
    if args.agg == "pct":
        max_t = 1
        min_t = rt
        fix_rec = 1
        fix_prec = 0
        while abs(fix_rec - fix_prec) > 0.0001:
            rt = (max_t + min_t) / 2
            verbose_print(args, f"---------- finding rt: {rt} ----------")

            _,_,_,ans, fix_prec, fix_rec = test_SPRinT(args, oracle_dist_S, proxy_dist_S, rt)

            if fix_prec < fix_rec:
                min_t = rt
            else:
                max_t = rt

            if abs(min_t - max_t) < 0.0001:
                verbose_print(args, "CANNOT find optimal rt!!")
                CANNOT = 1
                break
            verbose_print(
                args,
                f"rt = {rt}: Found NN: {len(ans)}, Pilot Precision {fix_prec}, Pilot Recall {fix_rec}",
            )

        optimal_t = rt
        verbose_print(
            args,
            f"Found optimal recall target={optimal_t} with fix prec {fix_prec} and fix rec {fix_rec}",
        )

    # F1 score
    else:
        left = 0
        right = 1
        tolerance = 0.01
        while right - left > tolerance:
            m1 = left + (right - left) / 3
            m2 = right - (right - left) / 3
            verbose_print(args, f"---------- finding rt between {m1, m2} ----------")
            f1_m1 = f1_score(args, m1, oracle_dist_S, proxy_dist_S)
            f1_m2 = f1_score(args, m2, oracle_dist_S, proxy_dist_S)

            if f1_m1 <= f1_m2:
                left = m1
            else:
                right = m2
        optimal_t = (left + right) / 2
        max_f1 = f1_score(args, optimal_t, oracle_dist_S, proxy_dist_S)

        # # naive brute force method
        # max_f1 = 0
        # f1_list = []
        # f_p=[]
        # f_r=[]
        # while rt <= 1:
        #     verbose_print(args, f"---------- finding rt: {rt} ----------")
        #     f1 = f1_score(args, rt, oracle_dist_S, proxy_dist_S)
        #     verbose_print(args, f"fix sample F1 score: {f1}")
        #     f1_list.append(f1)
        #
        #     if f1 > max_f1 - 0.01:
        #         max_f1 = f1
        #         optimal_t = rt
        #     rt += 0.01
        #
        # df = pd.DataFrame(f1_list, columns=["F1 score"])
        # df.to_csv(f"{args.agg}_{args.Fname}_{args.file_suffix}.csv", index=False)
        verbose_print(
            args,
            f"Found optimal recall target={optimal_t} with fix f1 {max_f1}",
        )
    return optimal_t, CANNOT


if __name__ == "__main__":
    # Parse arguments and initialize
    args = parse_args()
    args.start_time = time.time()

    # Load data
    Proxy_emb, Oracle_emb = load_data(args)

    args.fac_list = np.arange(
        float(args.fac_list.split(",")[0]),
        float(args.fac_list.split(",")[1]),
        float(args.fac_list.split(",")[2]),
    )

    verbose_print(args, f"Prob: {args.Prob}; r: {args.Dist_t}; beta: {args.beta}")
    verbose_print(
        args,
        f"Dataset: {args.Fname}, size: {Proxy_emb.shape} , aggregation function: {args.agg}, filename: {args.file_suffix}, s: {args.s}, s_p: {args.s_p}",
    )

    # Handle hypothesis-specific setup
    if args.agg == "pct":
        args.attr = "proportion"
    else:
        if args.Fname in ["eICU", "MIMIC-III"]:
            attr_filename = f"data/Medical/{args.Fname}/{args.Fname}.testfull"
            args.ori_D_attr = get_data(filename=attr_filename)
        elif args.Fname in ["Amazon-HH", "Amazon-E"]:
            attr_filename = f"data/Amazon/{args.Fname}/{args.Fname}.testfull"
            args.ori_D_attr = get_data(filename=attr_filename)

        elif args.Fname in ["Jackson"]:
            attr_filename = f"data/Video/jackson10000_attribute.csv"
            df = pd.read_csv(attr_filename)
            args.ori_D_attr = np.vstack(np.array(df["attribute_list"])).tolist()
            args.ori_D_attr = [["", "", speed] for speed in args.ori_D_attr]

        verbose_print(
            args,
            f"H1: {args.agg} {args.attr} (ind = {args.attr_id}) of NNs of q is tested.",
        )

    # Prepare output path
    Path(f"./results/{args.algo}/").mkdir(parents=True, exist_ok=True)

    for seed in range(1, 11):
        args.optimal_cost = None
        np.random.seed(seed)

        verbose_print(
            args, f"*********************** start seed {seed} ***********************"
        )

        # get oracle ground truth NN and agg
        query_indices = np.random.choice(
            range(len(Oracle_emb)), size=args.num_query, replace=False
        )
        query_emb = Oracle_emb[query_indices]
        Oracle_dist, Proxy_dist = prepare_distances(
            args, Oracle_emb, Proxy_emb, query_emb
        )

        args.true_ans_D = np.where(Oracle_dist <= args.Dist_t)[0]
        if args.agg == "pct":
            args.agg_D = len(args.true_ans_D) / Oracle_dist.shape[0]
            verbose_print(
                args,
                f"Ground Truth NN is {len(args.true_ans_D)}, proportion is {args.agg_D}",
            )
        else:
            args.l_D, args.agg_D = agg_value(
                args.ori_D_attr, args.true_ans_D, args.attr_id, "avg"
            )

            l_D_full, agg_D_full = agg_value(
                args.ori_D_attr, range(len(args.ori_D_attr)), args.attr_id, "avg"
            )

            half_size = len(args.true_ans_D) // 2
            std_dev = np.sqrt(np.var(args.l_D))
            bootstrapped_samples_1 = np.random.choice(args.l_D, half_size, replace=True)
            bootstrapped_samples_2 = np.random.choice(
                l_D_full, len(args.true_ans_D) - half_size, replace=True
            )

            noise_1 = np.random.uniform(-0.5 * std_dev, 0.5 * std_dev, size=half_size)
            noise_2 = np.random.uniform(
                -1.5 * std_dev, 1.5 * std_dev, size=len(args.true_ans_D) - half_size
            )

            # Apply noise to boost variance
            modified_distribution_1 = bootstrapped_samples_1 / 2 + noise_1
            modified_distribution_2 = bootstrapped_samples_2 + noise_2

            distribution = np.concatenate(
                [modified_distribution_1, modified_distribution_2]
            )

            if args.Fname in ["eICU"]:
                distribution = np.clip(distribution, 0, 100)

            elif args.Fname in ["MIMIC-III"]:
                distribution = np.clip(distribution, 30, 180)

            elif args.Fname in ["Amazon-HH", "Amazon-E"]:
                distribution = np.clip(distribution, 0, 5)

            elif args.Fname in ["Jackson"]:
                distribution = np.clip(distribution, 20, 100)

            np.random.shuffle(distribution)  # Shuffle to avoid order bias

            args.D_attr = copy.deepcopy(args.ori_D_attr)

            count = 0
            for idx in args.true_ans_D:
                args.D_attr[idx][2][args.attr_id] = distribution[count]
                count += 1

            args.l_D, args.agg_D = agg_value(
                args.D_attr, args.true_ans_D, args.attr_id, args.agg
            )
            print(args.agg_D)

            verbose_print(
                args,
                f"Ground Truth NN is {len(args.true_ans_D)} ({round(len(args.true_ans_D)/len(Oracle_emb),4)*100}%), aggregation is {args.agg_D}",
            )

        # rerun algo for num_sample times for average results
        (
            avg_execution_time,
            avg_error,
            avg_absError,
            avg_acc,
            avg_rec,
            avg_prec,
            avg_fix_rec,
            avg_fix_prec,
            avg_NN_RT,
            avg_agg,
            var_agg,
            avg_NN_S,
            avg_agg_S,
            standard_error,
            avg_CI,
            avg_f1,
            avg_fix_f1,
            cannot_times,
        ) = run_experiment(
            args,
            Oracle_dist,
            Proxy_dist,
            seed,
        )

        # output results
        output_results(
            args,
            seed,
            avg_execution_time,
            avg_error,
            avg_absError,
            avg_NN_RT,
            avg_agg,
            var_agg,
            avg_NN_S,
            avg_agg_S,
            standard_error,
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
