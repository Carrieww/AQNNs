import time
import numpy as np
from math import floor
from pathlib import Path
from scipy.stats import norm
from config import parse_args
from hyper_parameter import norm_scale
from aquapro_util import (
    prepare_distances,
    agg_value,
    get_data,
    load_data,
    preprocess_topk_phi,
    verbose_print,
    array_union,
    set_diff,
)
from HT_test import HT_acc_t_test, one_proportion_z_test, HT_acc_z_test

# from numba import njit


# @njit
def test_PQA_RT(
    args,
    oracle_dist,
    phi,
    topk,
    rt=0.9,
):
    true_ans = np.where(oracle_dist <= args.Dist_t)[0]
    if len(true_ans) == 0:
        return 0, 0, len(oracle_dist), np.empty(0, dtype=np.int64), None, None

    L = 1
    R = len(phi)

    def pb_distribution(phii, p):
        for j in range(1, len(phii) + 1):
            if j == 1:
                p[0] = 1 - phii[0]
                p[1] = phii[0]
            else:
                shift_p = np.roll(p, 1) * phii[j - 1]
                p = p * (1 - phii[j - 1]) + shift_p

        return p

    while L < R:
        mid = floor((L + R) / 2)

        pbs = pb_distribution(phi[:mid], np.zeros(len(phi) + 1))
        pbc = pb_distribution(phi[mid:], np.zeros(len(phi) + 1))

        recall_prob = 0
        for i in range(mid + 1):
            cdf = np.sum(pbc[: floor((1 - rt) * i / rt) + 1])
            recall_prob += pbs[i] * cdf

        if recall_prob < args.Prob:
            L = mid + 1
        else:
            R = mid

    k_star = L
    max_exp = 0
    pbs = np.zeros(len(phi) + 1)

    for i in range(L, len(phi) + 1):
        if i == L:
            pbs = pb_distribution(phi[:L], np.zeros(len(phi) + 1))
        else:
            shift_pbs = np.roll(pbs, 1) * phi[i - 1]
            pbs = pbs * (1 - phi[i - 1]) + shift_pbs

        exp_precis = np.sum(np.array([pbs[j] * j / i for j in range(i + 1)]))
        if exp_precis >= max_exp:
            k_star = i
            max_exp = exp_precis

    if args.samples is None:
        ans = topk[:k_star]
    else:
        pilots_false = args.samples[
            np.where(oracle_dist[args.samples] > args.Dist_t)[0]
        ]
        ans = set_diff(array_union(topk[:k_star], args.samples), pilots_false)

    # fixed sample's precision
    fix_true_ans = args.fix_sample[
        np.where(oracle_dist[args.fix_sample] <= args.Dist_t)[0]
    ]
    fix_true_pos = len(np.intersect1d(ans, fix_true_ans))
    if len(np.intersect1d(ans, args.fix_sample)) == 0:
        fix_precision = 0
    else:
        fix_precision = fix_true_pos / len(np.intersect1d(ans, args.fix_sample))
    if len(fix_true_ans) == 0:
        fix_recall = 0
    else:
        fix_recall = fix_true_pos / len(fix_true_ans)

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, k_star, ans, fix_precision, fix_recall


def test_PQE_RT(args, oracle_dist, proxy_dist, rt):
    topk, phi = preprocess_topk_phi(
        proxy_dist, norm_scale=args.est_scale, t=args.Dist_t
    )

    precision, recall, _, ans, fix_prec, fix_rec = test_PQA_RT(
        args,
        oracle_dist,
        phi,
        topk,
        rt=rt,
    )
    verbose_print(
        args,
        f"true precision is {precision}, true recall is {recall}, and the precision for fixed sample is {fix_prec}, and recall for fixed sample is {fix_rec}",
    )

    return precision, recall, _, ans, fix_prec, fix_rec


def PQE_better(
    args,
    Oracle_dist,
    seed,
):
    acc_l = []
    recall_l = []
    precision_l = []
    fix_prec_l = []
    fix_rec_l = []
    agg_l = []
    agg_S_l = []
    NN_S_l = []
    NN_RT_l = []
    cannot_times_l = []

    if args.hypothesis_type == "NNH":
        CI_l = []
        f1_l = []
        fix_f1_l = []

    for i in range(args.num_sample):
        np.random.seed(seed * i)

        indices = np.random.choice(Oracle_dist.shape[0], args.total_cost, replace=False)
        oracle_dist_S = Oracle_dist[indices]
        proxy_dist_S = Proxy_dist[indices]
        args.true_ans_S = np.where(oracle_dist_S <= args.Dist_t)[0]
        args.NN_S = len(args.true_ans_S)
        verbose_print(args, f"Find NN in S is {args.NN_S}")

        if args.hypothesis_type == "P-NNH":
            args.agg_S = len(args.true_ans_S) / oracle_dist_S.shape[0]
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
                f"The number of NN in S is {len(args.true_ans_S)} ({len(args.true_ans_S)/proxy_dist_S.shape[0]}%), the aggregation of true NN is {args.agg_S} and the aggregated value of all data in S is {args.agg_S_full}",
            )

        # prepare pilot sample
        args.fix_sample = np.random.choice(
            len(oracle_dist_S), size=int(args.initial_cost), replace=False
        )
        available_indices = np.setdiff1d(np.arange(len(oracle_dist_S)), args.fix_sample)
        args.available_oracle_dist_S = oracle_dist_S[available_indices]
        args.available_proxy_dist_S = proxy_dist_S[available_indices]

        # no probe sample
        args.est_scale = norm_scale
        args.samples = None

        # find optimal rt
        args.rt, CANNOT = find_optimal_rt(args, oracle_dist_S, proxy_dist_S)
        args.optimal_cost = args.initial_cost  # + args.probe_cost

        # find NN by SPRinT
        RT_precision, RT_recall, _, RT_ans, fix_prec, fix_rec = test_PQE_RT(
            args, oracle_dist_S, proxy_dist_S, args.rt
        )
        args.NN_RT = len(RT_ans)
        verbose_print(
            args,
            f"recall: {RT_recall}, prcision: {RT_precision} at cost {args.optimal_cost}, fix prec: {fix_prec}, fix recall: {fix_rec}",
        )

        recall_l.append(RT_recall)
        precision_l.append(RT_precision)
        fix_prec_l.append(fix_prec)
        fix_rec_l.append(fix_rec)

        if args.hypothesis_type == "P-NNH":
            approx_agg_S = round(len(RT_ans) / proxy_dist_S.shape[0], 4)
            verbose_print(args, f"the prop is {approx_agg_S}")
        elif args.hypothesis_type == "NNH":
            approx_l_S, approx_agg_S = agg_value(
                args.S_attr, RT_ans, args.attr_id, args.agg
            )
            verbose_print(
                args,
                f"The number of NN by SPRinT is {len(RT_ans)} ({len(RT_ans)/proxy_dist_S.shape[0]}%), the approximated aggregated value is {approx_agg_S}",
            )
            if (RT_precision + RT_recall) == 0:
                approx_f1 = 0
            else:
                approx_f1 = 2 * RT_recall * RT_precision / (RT_precision + RT_recall)
            if (fix_prec + fix_rec) == 0:
                approx_fix_f1 = 0
            else:
                approx_fix_f1 = 2 * fix_prec * fix_rec / (fix_prec + fix_rec)

        for fac in args.fac_list:
            for H1_op in ["greater", "less"]:
                if args.hypothesis_type == "P-NNH":
                    c_time_GT = (len(args.true_ans_D) / Oracle_dist.shape[0]) * fac
                    verbose_print(args, f">>> c is {c_time_GT}")

                    _, _, GT = one_proportion_z_test(
                        len(args.true_ans_D),
                        Oracle_dist.shape[0],
                        c_time_GT,
                        0.05,
                        H1_op,
                    )
                    verbose_print(
                        args, f"the ground truth to reject H0 result is : {GT}"
                    )

                    rt_align, _ = HT_acc_z_test(
                        args,
                        "PQE-RT",
                        RT_ans,
                        oracle_dist_S.shape[0],
                        GT,
                        c_time_GT,
                        H1_op,
                    )
                    acc_l.append(rt_align)

                elif args.hypothesis_type == "NNH":
                    c_time_GT = args.agg_D * fac
                    verbose_print(args, f">>> c is {c_time_GT}")

                    _, GT, _, _ = HT_acc_t_test(
                        args, args.l_D, c_time_GT, H1_op, is_D=True
                    )
                    verbose_print(
                        args, f"the ground truth to reject H0 result is : {GT}"
                    )

                    rt_align, _, rt_CI_l, rt_CI_h = HT_acc_t_test(
                        args, approx_l_S, c_time_GT, H1_op, GT=GT, is_D=False
                    )

                    acc_l.append(rt_align)
                    CI_l.append(rt_CI_h - rt_CI_l)
                    f1_l.append(approx_f1)
                    fix_f1_l.append(approx_fix_f1)

        agg_l.append(approx_agg_S)
        agg_S_l.append(args.agg_S)
        NN_S_l.append(args.NN_S)
        NN_RT_l.append(args.NN_RT)
        cannot_times_l.append(CANNOT)

    avg_acc = np.nanmean(acc_l)
    avg_recall = np.nanmean(recall_l)
    avg_precision = np.nanmean(precision_l)
    avg_fix_rec = np.nanmean(fix_rec_l)
    avg_fix_prec = np.nanmean(fix_prec_l)
    avg_NN_S = np.nanmean(NN_S_l)
    avg_NN_RT = np.nanmean(NN_RT_l)
    cannot_times = np.nanmean(cannot_times_l)
    verbose_print(
        args,
        f"the average accuracy over {args.num_sample} runs and {args.fac_list} is {avg_acc}",
    )
    verbose_print(args, f"the average recall over {args.num_sample} is {avg_recall}")
    verbose_print(
        args, f"the average precision over {args.num_sample} is {avg_precision}"
    )
    verbose_print(
        args, f"the average fix recall over {args.num_sample} is {avg_fix_rec}"
    )
    verbose_print(
        args, f"the average fix precision over {args.num_sample} is {avg_fix_prec}"
    )
    verbose_print(args, f"the average NN in S over {args.num_sample} is {avg_NN_S}")
    verbose_print(
        args, f"the average NN by SPRinT in S over {args.num_sample} is {avg_NN_RT}"
    )

    avg_agg = np.nanmean(agg_l)
    var_agg = np.nanvar(agg_l)
    avg_agg_S = np.nanmean(agg_S_l)
    var_agg_S = np.nanvar(agg_S_l)

    if args.hypothesis_type == "P-NNH":
        verbose_print(
            args,
            f"the average prop_S over {args.num_sample} is {avg_agg} with variance {round(var_agg,4)}",
        )
        return (
            avg_acc,
            avg_recall,
            avg_precision,
            avg_fix_rec,
            avg_fix_prec,
            avg_NN_RT,
            avg_agg,
            var_agg,
            avg_NN_S,
            avg_agg_S,
            var_agg_S,
            None,
            None,
            None,
            cannot_times,
        )
    elif args.hypothesis_type == "NNH":
        avg_CI = np.mean(CI_l)
        avg_f1 = np.mean(f1_l)
        avg_fix_f1 = np.mean(fix_f1_l)
        verbose_print(
            args,
            f"the average aggregate value over {args.num_sample} is {avg_agg} with variance {round(var_agg,4)}",
        )
        verbose_print(
            args,
            f"the average aggregate value in S over {args.num_sample} is {avg_agg_S} with variance {round(var_agg_S,4)}",
        )
        verbose_print(args, f"the average CI over {args.num_sample} is {avg_CI}")
        verbose_print(args, f"the average f1 score over {args.num_sample} is {avg_f1}")
        verbose_print(
            args, f"the average fix f1 score over {args.num_sample} is {avg_fix_f1}"
        )
        return (
            avg_acc,
            avg_recall,
            avg_precision,
            avg_fix_rec,
            avg_fix_prec,
            avg_NN_RT,
            avg_agg,
            var_agg,
            avg_NN_S,
            avg_agg_S,
            var_agg_S,
            avg_CI,
            avg_f1,
            avg_fix_f1,
            None,
        )


def process_results_NNH(
    args,
    seed,
    avg_NN_RT,
    avg_agg,
    var_agg,
    avg_NN_S,
    avg_agg_S,
    var_agg_S,
    avg_CI,
    avg_f1,
    avg_fix_f1,
    avg_acc,
    avg_recall,
    avg_precision,
    avg_fix_rec,
    avg_fix_prec,
    cannot_times,
):
    # Saving result
    file_name = f"results/SPRinT_{args.PQA}/{args.hypothesis_type}_{args.Fname}_{args.version}.txt"
    with open(file_name, "a") as file:
        # Write the header (if it's not already present in the file)
        if seed == 1:
            file.write(
                "seed\toptimal cost\tno optimal rt counts\tavg acc\tavg recall\tavg precision\tavg f1\tavg fix recall\tavg fix precision\tavg fix f1\tagg ours\tvar ours\tNN ours\tagg_S\tvar agg_S\tNN S\tavg CI\n"
            )

        # Write the data for the current seed
        if avg_CI is None:  # P-NNH
            file.write(
                f"{seed:.4f}\t{args.optimal_cost:.4f}\t{cannot_times:.4f}\t{avg_acc:.4f}\t{avg_recall:.4f}\t{avg_precision:.4f}\t{avg_f1}\t{avg_fix_rec:.4f}\t{avg_fix_prec:.4f}\t{avg_fix_f1}\t{avg_agg:.4f}\t{var_agg:.4f}\t{avg_NN_RT:.4f}\t{avg_agg_S}\t{var_agg_S}\t{avg_NN_S:.4f}\t{None}\n"
            )
        else:
            file.write(
                f"{seed:.4f}\t{args.optimal_cost:.4f}\t{None}\t{avg_acc:.4f}\t{avg_recall:.4f}\t{avg_precision:.4f}\t{avg_f1:.4f}\t{avg_fix_rec:.4f}\t{avg_fix_prec:.4f}\t{avg_fix_f1:.4f}\t{avg_agg:.4f}\t{var_agg:.4f}\t{avg_NN_RT:.4f}\t{avg_agg_S:.4f}\t{var_agg_S:.4f}\t{avg_NN_S:.4f}\t{avg_CI:.4f}\n"
            )
    verbose_print(
        args,
        f"At recall target={args.recall_target} and precision target={args.precision_target}; we find optimal cost={args.optimal_cost} and optimal rt={args.rt}",
    )
    verbose_print(
        args,
        f"avg acc: {avg_acc}, avg recall: {avg_recall}, avg precision: {avg_precision}, avg fix recall: {avg_fix_rec}, avg precision: {avg_fix_prec}",
    )
    verbose_print(args, "execution time is %.2fs" % (time.time() - args.start_time))


def find_optimal_rt(args, oracle_dist_S, proxy_dist_S, rt=0.1):
    CANNOT = 0
    if args.hypothesis_type == "P-NNH":
        max_t = 1
        min_t = rt
        RT_fix_rec = 1
        RT_fix_prec = 0
        while abs(RT_fix_rec - RT_fix_prec) > 0.01:
            rt = (max_t + min_t) / 2
            verbose_print(args, f"---------- finding rt: {rt} ----------")
            (
                RT_precision,
                RT_recall,
                _,
                RT_ans,
                RT_fix_prec,
                RT_fix_rec,
            ) = test_PQE_RT(args, oracle_dist_S, proxy_dist_S, rt)
            if RT_fix_rec < RT_fix_prec:
                min_t = rt
            else:
                max_t = rt

            if abs(min_t - max_t) < 0.001:
                verbose_print(args, "CANNOT find optimal rt!!")
                CANNOT = 1
                break
            verbose_print(
                args,
                f"rt = {rt}: Found NN: {len(RT_ans)}, Precision {RT_precision}, Recall {RT_recall}, Pilot Precision {RT_fix_prec}, Pilot Recall {RT_fix_rec}",
            )

        optimal_t = rt

    # F1 score
    elif args.hypothesis_type == "NNH":
        max_f1 = 0
        while rt <= 1:
            verbose_print(args, f"---------- finding rt: {rt} ----------")
            (
                RT_precision,
                RT_recall,
                _,
                RT_ans,
                RT_fix_prec,
                RT_fix_rec,
            ) = test_PQE_RT(args, oracle_dist_S, proxy_dist_S, rt)
            if (RT_fix_prec + RT_fix_rec) == 0:
                RT_F1 = 0
            else:
                RT_F1 = RT_fix_prec * RT_fix_rec * 2 / (RT_fix_prec + RT_fix_rec)
            verbose_print(args, f"Found NN: {len(RT_ans)}, with F1 score: {RT_F1}")

            if RT_F1 >= max_f1:
                max_f1 = RT_F1
                optimal_t = rt
            rt += 0.01

    verbose_print(
        args,
        f"Found optimal recall target={optimal_t}, we achieve recall {RT_recall} and precision: {RT_precision}, fix precision: {RT_fix_prec} and fix recall: {RT_fix_rec}",
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

    verbose_print(args, f"Prob: {args.Prob}; r: {args.Dist_t}")
    verbose_print(
        args,
        f"Dataset: {args.Fname}, method: {args.hypothesis_type}, assumption: {args.PQA}, version: {args.version}, cost: {args.total_cost}",
    )

    # Handle hypothesis-specific setup
    if args.hypothesis_type == "NNH":
        args.D_attr = get_data(
            filename=f"data/medical/{args.Fname}/{args.Fname}.testfull"
        )
        args.agg = "mean"
        verbose_print(args, f"H1: {args.agg} {args.attr} of NNs of q is tested.")
    elif args.hypothesis_type == "P-NNH":
        args.attr = "proportion"

    # Prepare output path
    Path(f"./results/SPRinT_{args.PQA}/").mkdir(parents=True, exist_ok=True)

    for seed in range(1, 2):
        args.optimal_cost = None
        np.random.seed(seed)

        verbose_print(
            args, f"*********************** start seed {seed} ***********************"
        )
        # get oracle ground truth NN and agg
        query_indices = np.random.choice(
            range(len(Oracle_emb)), size=args.num_query, replace=False
        )
        Oracle_dist, Proxy_dist = prepare_distances(
            args, Oracle_emb, Proxy_emb, query_indices
        )

        args.true_ans_D = np.where(Oracle_dist <= args.Dist_t)[0]
        if args.hypothesis_type == "NNH":
            args.l_D, args.agg_D = agg_value(
                args.D_attr, args.true_ans_D, args.attr_id, args.agg
            )
            verbose_print(args, f"Ground Truth Aggregation: {args.agg_D}")
        elif args.hypothesis_type == "P-NNH":
            args.prop_D = len(args.true_ans_D) / Oracle_dist.shape[0]
            verbose_print(args, f"Ground Truth Proportion: {args.prop_D}")

        # rerun algo for 30 times for average results
        (
            avg_acc,
            avg_recall,
            avg_precision,
            avg_fix_recall,
            avg_fix_precision,
            avg_NN_RT,
            avg_agg,
            var_agg,
            avg_NN_S,
            avg_agg_S,
            var_agg_S,
            avg_CI,
            avg_f1,
            avg_fix_f1,
            cannot_times,
        ) = PQE_better(
            args,
            Oracle_dist,
            seed,
        )

        # output results
        process_results_NNH(
            args,
            seed,
            avg_NN_RT,
            avg_agg,
            var_agg,
            avg_NN_S,
            avg_agg_S,
            var_agg_S,
            avg_CI,
            avg_f1,
            avg_fix_f1,
            avg_acc,
            avg_recall,
            avg_precision,
            avg_fix_recall,
            avg_fix_precision,
            cannot_times,
        )
