from math import floor, ceil
from scipy import stats
from config import parse_args

from aquapro_util import (
    load_data,
    preprocess_dist,
    preprocess_topk_phi,
    preprocess_ranks,
)
from aquapro_util import array_union, set_diff, preprocess_sync

from numba import njit
from hyper_parameter import std_offset, norm_scale
import numpy as np

import pickle
from pathlib import Path

import time

#
# @njit
# def test_PQA_PT(oracle_dist, phi, topk, t=0.9, prob=0.9, pt=0.9, pilots=None):
#     true_ans = np.where(oracle_dist <= t)[0]
#     if len(true_ans) == 0:
#         return 0, 0, 0, np.empty(0, dtype=np.int64)
#
#     pbs = np.zeros(len(phi) + 1)
#     k_star = 0
#
#     for i in range(1, len(phi) + 1):
#         if i == 1:
#             pbs[0] = 1 - phi[0]
#             pbs[1] = phi[0]
#         else:
#             shift_pbs = np.roll(pbs, 1) * phi[i - 1]
#             pbs = pbs * (1 - phi[i - 1]) + shift_pbs
#
#         idx_s = ceil(i * pt)
#         precis_prob = np.sum(pbs[idx_s : i + 1])
#
#         if precis_prob >= prob:
#             k_star = i
#
#     if k_star == 0:
#         return 0, 0, 0, np.empty(0, dtype=np.int64)
#
#     if pilots is None:
#         ans = topk[:k_star]
#     else:
#         pilots_false = pilots[np.where(oracle_dist[pilots] > t)[0]]
#         ans = set_diff(array_union(topk[:k_star], pilots), pilots_false)
#
#     true_pos = len(np.intersect1d(ans, true_ans))
#     precision = true_pos / len(ans)
#     recall = true_pos / len(true_ans)
#
#     return precision, recall, k_star, ans


# @njit
def test_PQA_RT(
    oracle_dist,
    phi,
    topk,
    fix_sample,
    t=0.9,
    prob=0.9,
    rt=0.9,
    pilots=None,
):
    true_ans = np.where(oracle_dist <= t)[0]
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

        if recall_prob < prob:
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

    if pilots is None:
        ans = topk[:k_star]
    else:
        pilots_false = pilots[np.where(oracle_dist[pilots] > t)[0]]
        ans = set_diff(array_union(topk[:k_star], pilots), pilots_false)

    # fixed sample's precision
    fix_true_ans = fix_sample[np.where(oracle_dist[fix_sample] <= t)[0]]
    fix_true_pos = len(np.intersect1d(ans, fix_true_ans))
    if len(np.intersect1d(ans, fix_sample)) == 0:
        fix_precision = 0
    else:
        fix_precision = fix_true_pos / len(np.intersect1d(ans, fix_sample))
    if len(fix_true_ans) == 0:
        fix_recall = 0
    else:
        fix_recall = fix_true_pos / len(fix_true_ans)

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, k_star, ans, fix_precision, fix_recall


# def test_PQE_PT(oracle_dist, proxy_dist, bd, t=0.9, prob=0.9, pt=0.9):
#     imp_p = (1 - proxy_dist + 1e-3) / np.sum(1 - proxy_dist + 1e-3)
#     samples = np.random.choice(len(oracle_dist), size=bd, replace=False, p=imp_p)
#
#     est_scale = np.std(oracle_dist[samples] - proxy_dist[samples]) + std_offset
#     print(f"norm scale is {est_scale}")
#
#     topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=est_scale, t=t)
#
#     precision, recall, _, ans = test_PQA_PT(
#         oracle_dist, phi, topk, t=t, prob=prob, pt=pt, pilots=samples
#     )
#
#     return precision, recall, _, ans


def test_PQE_RT(args, oracle_dist, proxy_dist, rt, bd):
    if bd - len(args.fix_sample) == 0:
        topk, phi = preprocess_topk_phi(
            proxy_dist, norm_scale=norm_scale, t=args.Dist_t
        )
        samples = None
    else:
        imp_p = (1 - args.available_proxy_dist_S + 1e-3) / np.sum(
            1 - args.available_proxy_dist_S + 1e-3
        )
        samples = np.random.choice(
            len(args.available_oracle_dist_S),
            size=bd - len(args.fix_sample),
            replace=False,
            p=imp_p,
        )

        est_scale = (
            np.std(
                args.available_oracle_dist_S[samples]
                - args.available_proxy_dist_S[samples]
            )
            + std_offset
        )

        topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=est_scale, t=args.Dist_t)

    precision, recall, _, ans, fix_prec, fix_rec = test_PQA_RT(
        oracle_dist,
        phi,
        topk,
        args.fix_sample,
        t=args.Dist_t,
        prob=args.Prob,
        rt=rt,
        pilots=samples,
    )
    print(
        f"true precision is {precision}, true recall is {recall}, and the precision for fixed sample is {fix_prec}, and recall for fixed sample is {fix_rec}"
    )

    return precision, recall, _, ans, fix_prec, fix_rec


def HT_acc_z_test(args, name, ans, total, GT, prop_c):
    if args.verbose:
        print(f"finished {name} algorithm for q")
        print(f"FRNN result: {len(ans)}")
        print(f"total patients: {total}")
        print(f"c proportion: {prop_c}; approx: {len(ans) / total}")

    z_stat, p_value, reject = one_proportion_z_test(
        len(ans), total, prop_c, 0.05, args.H1_op
    )

    if args.verbose:
        print("Z-Statistic:", z_stat)
        print("P-Value:", p_value)
        print("Reject Null Hypothesis:", reject)

    align = reject == GT

    if args.verbose:
        print("align:", align)

    return align, reject


from scipy.stats import norm


def one_proportion_z_test(
    successes, total_trials, null_prop, alpha=0.05, alternative="two-sided"
):
    """
    Perform a one-proportion z-test.

    Parameters:
    - successes: Number of successes.
    - total_trials: Total number of trials.
    - null_prop: The hypothesized population proportion under the null hypothesis.
    - alpha: Significance level (default is 0.05).
    - alternative: The alternative hypothesis ('two-sided', 'less', or 'greater'). Default is 'two-sided'.

    Returns:
    - z_stat: The z-statistic.
    - p_value: The p-value.
    - rejection: True if the null hypothesis is rejected, False otherwise.
    """

    # Calculate sample proportion
    sample_prop = successes / total_trials

    # Calculate standard error
    std_error = (null_prop * (1 - null_prop) / total_trials) ** 0.5

    # Calculate z-statistic
    z_stat = (sample_prop - null_prop) / std_error

    # Calculate p-value
    if alternative == "two-sided":
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    elif alternative == "less":
        p_value = norm.cdf(z_stat)
    elif alternative == "greater":
        p_value = 1 - norm.cdf(z_stat)

    # Determine rejection of null hypothesis
    reject = p_value < alpha

    return z_stat, p_value, reject


def PQE_better(
    args,
    Oracle_dist,
    oracle_dist_S,
    proxy_dist_S,
    seed,
):
    acc_l = []
    recall_l = []
    precision_l = []
    fix_prec_l = []
    fix_rec_l = []
    agg_l = []

    if args.method == "NNH":
        CI_l_l = []
        CI_h_l = []
        f1_l = []
        fix_f1_l = []

    for i in range(args.num_sample):
        np.random.seed(seed * i)
        # if args.PQA =="PQA":

        #     indices = np.random.choice(Oracle_dist.shape[0], args.total_cost, replace=False)
        #     oracle_dist_S = Oracle_dist[indices]
        #     proxy_dist_S = Proxy_dist[indices]

        RT_precision, RT_recall, _, RT_ans, fix_prec, fix_rec = test_PQE_RT(
            args, oracle_dist_S, proxy_dist_S, args.rt, args.optimal_cost
        )
        print(
            f"recall: {RT_recall}, prcision: {RT_precision}, at cost {args.optimal_cost}, fix prec: {fix_prec}, fix recall: {fix_rec}"
        )
        recall_l.append(RT_recall)
        precision_l.append(RT_precision)
        fix_prec_l.append(fix_prec)
        fix_rec_l.append(fix_rec)

        if args.method == "P-NNH":
            approx_prop_S = round(len(RT_ans) / proxy_dist_S.shape[0], 4)
            print(f"the prop is {approx_prop_S}")
        elif args.method == "NNH":
            approx_l_S, approx_agg_S = agg_value(
                args.S_attr, RT_ans, args.attr_id, args.agg
            )
            print(
                f"The number of NN in S is {len(RT_ans)} ({len(RT_ans)/proxy_dist_S.shape[0]}%), the approximated aggregated value is {approx_agg_S}"
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
            if args.method == "P-NNH":
                c_time_GT = (len(args.true_ans_D) / Oracle_dist.shape[0]) * fac
                if args.verbose:
                    print(f">>> c is {c_time_GT}")
                _, _, GT = one_proportion_z_test(
                    len(args.true_ans_D),
                    Oracle_dist.shape[0],
                    c_time_GT,
                    0.05,
                    args.H1_op,
                )
                if args.verbose:
                    print(f"the ground truth to reject H0 result is : {GT}")
                rt_align, _ = HT_acc_z_test(
                    args,
                    "PQE-RT",
                    RT_ans,
                    oracle_dist_S.shape[0],
                    GT,
                    c_time_GT,
                )
                acc_l.append(rt_align)
                agg_l.append(approx_prop_S)
            elif args.method == "NNH":
                c_time_GT = args.agg_D * fac
                print(f">>> c is {c_time_GT}")

                _, GT, _, _ = HT_acc_t_test(args.l_D, c_time_GT, args.H1_op, is_D=True)

                print(f"the ground truth to reject H0 result is : {GT}")
                rt_align, _, rt_CI_l, rt_CI_h = HT_acc_t_test(
                    approx_l_S, c_time_GT, args.H1_op, GT=GT, is_D=False
                )
                acc_l.append(rt_align)
                agg_l.append(approx_agg_S)
                CI_l_l.append(rt_CI_l)
                CI_h_l.append(rt_CI_h)
                f1_l.append(approx_f1)
                fix_f1_l.append(approx_fix_f1)

    avg_acc = np.nanmean(acc_l)
    avg_recall = np.nanmean(recall_l)
    avg_precision = np.nanmean(precision_l)
    avg_fix_rec = np.nanmean(fix_rec_l)
    avg_fix_prec = np.nanmean(fix_prec_l)
    avg_agg = np.nanmean(agg_l)
    print(
        f"the average accuracy over {args.num_sample} runs and {args.fac_list} is {avg_acc}"
    )
    print(f"the average recall over {args.num_sample} is {avg_recall}")
    print(f"the average precision over {args.num_sample} is {avg_precision}")
    print(f"the average fix recall over {args.num_sample} is {avg_fix_rec}")
    print(f"the average fix precision over {args.num_sample} is {avg_fix_prec}")

    if args.method == "P-NNH":
        print(f"the average prop_S over {args.num_sample} is {avg_agg}")
        return (
            avg_acc,
            avg_recall,
            avg_precision,
            avg_fix_rec,
            avg_fix_prec,
            avg_agg,
            None,
            None,
            None,
            None,
        )
    elif args.method == "NNH":
        avg_CI_l = np.mean(CI_l_l)
        avg_CI_h = np.mean(CI_h_l)
        avg_f1 = np.mean(f1_l)
        avg_fix_f1 = np.mean(fix_f1_l)
        print(f"the average aggregate value over {args.num_sample} is {avg_agg}")
        print(f"the average lower CI value over {args.num_sample} is {avg_CI_l}")
        print(f"the average upper CI value over {args.num_sample} is {avg_CI_h}")
        print(f"the average f1 score over {args.num_sample} is {avg_f1}")
        print(f"the average fix f1 score over {args.num_sample} is {avg_fix_f1}")
        return (
            avg_acc,
            avg_recall,
            avg_precision,
            avg_fix_rec,
            avg_fix_prec,
            avg_agg,
            avg_CI_l,
            avg_CI_h,
            avg_f1,
            avg_fix_f1,
        )


def get_avg_HT_acc(args, Oracle_dist, ans, oracle_dist_S):
    """
    Function to process the fac_list and return accuracy values.
    """
    acc_list = []

    for fac in args.fac_list:
        # Calculate c_time_GT based on fac
        c_time_GT = (len(args.true_ans_D) / Oracle_dist.shape[0]) * fac

        if args.verbose:
            print(f">>> c is {c_time_GT}")

        # Perform one proportion z-test
        _, _, GT = one_proportion_z_test(
            len(args.true_ans_D),
            Oracle_dist.shape[0],
            c_time_GT,
            0.05,
            args.H1_op,
        )

        if args.verbose:
            print(f"the ground truth to reject H0 result is : {GT}")

        # Calculate accuracy
        align, _ = HT_acc_z_test(
            args, "PQE-RT", ans, oracle_dist_S.shape[0], GT, c_time_GT
        )
        acc_list.append(align)

    return acc_list


def process_results_P_NNH(
    args,
    seed,
    find_cost_cost_list,
    find_cost_r_list,
    find_cost_p_list,
    find_cost_fix_r_list,
    find_cost_fix_p_list,
    find_cost_acc_list,
    find_cost_time,
    avg_acc,
    avg_recall,
    avg_precision,
):
    # Saving results
    file_name1 = f"results/SPRinT_{args.PQA}/{args.method}_{args.Fname}_{args.H1_op}_1016_{args.version}_costData.txt"
    with open(file_name1, "a") as file:
        file.write(f">>> seed {seed}\n")
        cost_str = "cost" + "\t" + "\t".join(map(str, find_cost_cost_list)) + "\n"
        file.write(cost_str)
        r_str = "recall" + "\t" + "\t".join(map(str, find_cost_r_list)) + "\n"
        file.write(r_str)
        fix_r_str = (
            "fix recall" + "\t" + "\t".join(map(str, find_cost_fix_r_list)) + "\n"
        )
        file.write(fix_r_str)
        p_str = "fix precision" + "\t" + "\t".join(map(str, find_cost_p_list)) + "\n"
        file.write(p_str)
        fix_p_str = (
            "fix precision" + "\t" + "\t".join(map(str, find_cost_fix_p_list)) + "\n"
        )
        file.write(fix_p_str)
        acc_str = "accuracy" + "\t" + "\t".join(map(str, find_cost_acc_list)) + "\n"
        file.write(acc_str)
        file.write("time" + "\t" + str(find_cost_time) + "\n\n")

    # file_name2 = f"results/SPRinT_{args.PQA}/{args.method}_{args.Fname}_{args.H1_op}_1016_{args.version}.txt"
    # with open(file_name2, "a") as file:
    #     if seed == 1:
    #         file.write(
    #             "seed\toptimal rt\toptimal cost\tavg acc\tavg recall\tavg precision\tGT proportion\n"
    #         )
    #     file.write(
    #         f"{seed:.4f}\t{args.rt:.4f}\t{args.optimal_cost:.4f}\t{avg_acc:.4f}\t{avg_recall:.4f}\t{avg_precision:.4f}\t{args.agg_D:.4f}\n"
    #     )
    # print(
    #     f"At recall target={args.recall_target} and precision target={args.precision_target}; we find optimal cost={args.optimal_cost} and optimal rt={args.rt}"
    # )
    # print(
    #     f"avg acc: {avg_acc}, avg recall: {avg_recall}, avg precision: {avg_precision}"
    # )
    end_time = time.time()
    print("execution time is %.2fs" % (end_time - args.start_time))


def process_results_NNH(
    args,
    seed,
    avg_agg,
    avg_CI_l,
    avg_CI_h,
    avg_f1,
    avg_fix_f1,
    avg_acc,
    avg_recall,
    avg_precision,
    avg_fix_rec,
    avg_fix_prec,
):
    # Saving result
    file_name = f"results/SPRinT_{args.PQA}/{args.method}_{args.Fname}_{args.H1_op}_1016_{args.version}.txt"
    with open(file_name, "a") as file:
        # Write the header (if it's not already present in the file)
        if seed == 1:
            file.write(
                "seed\toptimal rt\toptimal cost\tavg acc\tavg recall\tavg precision\tavg f1\tavg fix recall\tavg fix precision\tavg fix f1\tavg aggregation\tavg lower CI\tavg higher CI\tagg_D\tagg_D_full\tagg_S\tagg_S_full\n"
            )

        # Write the data for the current seed
        if avg_CI_l is None and avg_CI_h is None:  # P-NNH
            file.write(
                f"{seed:.4f}\t{args.rt:.4f}\t{args.optimal_cost:.4f}\t{avg_acc:.4f}\t{avg_recall:.4f}\t{avg_precision:.4f}\t{avg_f1}\t{avg_fix_rec:.4f}\t{avg_fix_prec:.4f}\t{avg_fix_f1}\t{avg_agg:.4f}\t{avg_CI_l}\t{avg_CI_h}\t{args.agg_D:.4f}\t{None}\t{args.agg_S:.4f}\t{None}\n"
            )
        else:
            file.write(
                f"{seed:.4f}\t{args.rt:.4f}\t{args.optimal_cost:.4f}\t{avg_acc:.4f}\t{avg_recall:.4f}\t{avg_precision:.4f}\t{avg_f1:.4f}\t{avg_fix_rec:.4f}\t{avg_fix_prec:.4f}\t{avg_fix_f1:.4f}\t{avg_agg:.4f}\t{avg_CI_l:.4f}\t{avg_CI_h:.4f}\t{args.agg_D:.4f}\t{args.agg_D_full:.4f}\t{args.agg_S:.4f}\t{args.agg_S_full:.4f}\n"
            )
    print(
        f"At recall target={args.recall_target} and precision target={args.precision_target}; we find optimal cost={args.optimal_cost} and optimal rt={args.rt}"
    )
    print(
        f"avg acc: {avg_acc}, avg recall: {avg_recall}, avg precision: {avg_precision}, avg fix recall: {avg_fix_rec}, avg precision: {avg_fix_prec}"
    )
    end_time = time.time()
    print("execution time is %.2fs" % (end_time - args.start_time))


def get_data(filename=None, is_text=False):
    if is_text:
        instance_list = (np.loadtxt(filename) >= 0).astype(int)
    else:
        instance_list = pickle.load(open(filename, "rb"), encoding="latin1")

    return instance_list


def is_int(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def agg_value(D, ind_list, attr_id, agg):
    l = []
    for ans_id in ind_list:
        value = D[ans_id][2][attr_id]
        if is_int(value):
            value = int(value)
            if not np.isnan(value):
                l.append(int(value))
        else:
            pass

    if agg == "mean":
        res = sum(l) / len(l)
    else:
        raise Exception(f"The case for {agg} has not been implemented yet")
    return l, res


def HT_acc_t_test(l, c, operator, GT=None, is_D=False):
    t_stat, p_value, rejectH0, CI_l, CI_h = one_sample_t_test(
        l, c, alternative=operator
    )

    print("T-Statistic:", t_stat)
    print("P-Value:", p_value)

    if is_D:
        align = True
        print(f"The ans in D to reject H0 result is : {rejectH0}")
        print("align with ground truth?", align)

    else:
        print(f"The ans to reject H0 result is : {rejectH0}")
        assert GT is not None, "GT is None"
        align = rejectH0 == GT
        print("align with ground truth?", align)

    return align, rejectH0, CI_l, CI_h


def one_sample_t_test(l, c, alpha=0.05, alternative="two-sided"):
    global rejectH0
    t_stat, p_value = stats.ttest_1samp(l, popmean=c, alternative=alternative)
    CI_lower, CI_upper = stats.t.interval(
        confidence=1 - alpha,
        df=len(l) - 1,
        loc=np.mean(l),
        scale=stats.sem(l),
    )

    if p_value < alpha:
        rejectH0 = True
        print(
            f"The test (c = {c}, op = {alternative}) is significant, we shall reject the null hypothesis."
        )
    elif p_value >= alpha:
        rejectH0 = False
        print(
            f"The test (c = {c}, op = {alternative}) is NOT significant, we shall accept the null hypothesis."
        )
    print(f"confidence interval is ({round(CI_lower,4), round(CI_upper,4)})")
    return t_stat, p_value, rejectH0, CI_lower, CI_upper


def find_optimal_cost(
    args,
    oracle_dist_S,
    proxy_dist_S,
    Oracle_dist,
):
    """
    Function to find the optimal cost based on the precision and recall targets.
    It iterates over increasing costs and checks the precision target to determine the optimal cost.
    """
    assert (
        args.initial_cost < args.total_cost
    ), "Initial cost must be less than total cost."

    cost = args.initial_cost
    optimal_cost = args.total_cost  # Default to total_cost if not found
    find_cost_flag = False

    # Initialize lists to store results
    find_cost_cost_list = []
    find_cost_r_list = []
    find_cost_p_list = []
    find_cost_fix_r_list = []
    find_cost_fix_p_list = []
    find_cost_acc_list = []

    # Iterate over cost increments to find optimal cost
    while cost <= args.total_cost:
        print(f"---------- finding cost: {cost} ----------")

        (
            RT_precision,
            RT_recall,
            RT_k_star,
            RT_ans,
            RT_fix_prec,
            RT_fix_rec,
        ) = test_PQE_RT(args, oracle_dist_S, proxy_dist_S, args.rt, cost)

        print(
            f"prop is {len(RT_ans)/len(oracle_dist_S)}, recall: {RT_recall}, precision: {RT_precision}, fix precision: {RT_fix_prec}, fix recall: {RT_fix_rec}"
        )

        # Append new results
        find_cost_r_list.append(round(RT_recall, 4))
        find_cost_p_list.append(round(RT_precision, 4))
        find_cost_fix_r_list.append(round(RT_fix_rec, 4))
        find_cost_fix_p_list.append(round(RT_fix_prec, 4))
        find_cost_cost_list.append(cost)

        # Recalculate accuracy
        acc_list = get_avg_HT_acc(args, Oracle_dist, RT_ans, oracle_dist_S)
        find_cost_acc_list.append(round(np.mean(acc_list), 4))

        # Check if we meet the precision target
        if RT_fix_prec >= args.precision_target and not find_cost_flag:
            print("FOUND OPTIMAL COST!")
            find_cost_flag = True
            optimal_cost = cost
        else:
            cost += args.cost_step_size

    if not find_cost_flag:
        print("DO NOT FIND OPTIMAL COST! Using total_cost.")

    print(
        f"At recall target={args.recall_target}; we find optimal cost={optimal_cost} which achieves recall {RT_recall}, precision {RT_precision}, fix prec {RT_fix_prec}, and HT accuracy {round(np.mean(acc_list), 4)}"
    )

    # Return the results and optimal cost
    return (
        optimal_cost,
        find_cost_cost_list,
        find_cost_r_list,
        find_cost_p_list,
        find_cost_fix_r_list,
        find_cost_fix_p_list,
        find_cost_acc_list,
        RT_precision,
        RT_recall,
    )


def find_optimal_rt(args, oracle_dist_S, proxy_dist_S, cost):
    rt = args.rt
    if args.PQA == "PQA" and args.method == "P-NNH":
        max_t = 1
        min_t = rt
        RT_fix_rec = 1
        RT_fix_prec = 0
        while abs(RT_fix_rec - RT_fix_prec) > 0.01:
            rt = (max_t + min_t) / 2
            print(f"---------- finding rt: {rt} ----------")
            (
                RT_precision,
                RT_recall,
                _,
                RT_ans,
                RT_fix_prec,
                RT_fix_rec,
            ) = test_PQE_RT(
                args,
                oracle_dist_S,
                proxy_dist_S,
                rt,
                cost,
            )
            if RT_fix_rec < RT_fix_prec:
                min_t = rt
            else:
                max_t = rt
        optimal_t = rt

    # elif args.PQA == "PQA" and args.method == "NNH":
    #     max_sum_p_r = 0
    #     while rt <= 1:
    #         print(f"---------- finding rt: {rt} ----------")
    #         (
    #             RT_precision,
    #             RT_recall,
    #             _,
    #             RT_ans,
    #             RT_fix_prec,
    #             RT_fix_rec,
    #         ) = test_PQE_RT(
    #             args,
    #             oracle_dist_S,
    #             proxy_dist_S,
    #             rt,
    #             cost,
    #         )
    #         print(f"Found NN: {len(RT_ans)}")
    #         if RT_fix_rec+RT_fix_prec >= max_sum_p_r:
    #             max_sum_p_r = RT_fix_rec + RT_fix_prec
    #             optimal_t = rt
    #         rt += 0.01

    # F1 score
    elif args.PQA == "PQA" and args.method == "NNH":
        max_f1 = 0
        while rt <= 1:
            print(f"---------- finding rt: {rt} ----------")
            (
                RT_precision,
                RT_recall,
                _,
                RT_ans,
                RT_fix_prec,
                RT_fix_rec,
            ) = test_PQE_RT(
                args,
                oracle_dist_S,
                proxy_dist_S,
                rt,
                cost,
            )
            if (RT_fix_prec + RT_fix_rec) == 0:
                RT_F1 = 0
            else:
                RT_F1 = RT_fix_prec * RT_fix_rec * 2 / (RT_fix_prec + RT_fix_rec)
            print(f"Found NN: {len(RT_ans)}, with F1 score: {RT_F1}")
            if RT_F1 >= max_f1:
                max_f1 = RT_F1
                optimal_t = rt
            rt += 0.01

    else:
        while rt <= 1:
            print(f"---------- finding rt: {rt} ----------")
            (
                RT_precision,
                RT_recall,
                _,
                RT_ans,
                RT_fix_prec,
                RT_fix_rec,
            ) = test_PQE_RT(
                args,
                oracle_dist_S,
                proxy_dist_S,
                rt,
                cost,
            )
            print(f"Found NN: {len(RT_ans)}")
            if RT_fix_rec >= args.recall_target:
                break
            else:
                rt += 0.01
        optimal_t = rt
    print(
        f"Found optimal recall target={optimal_t}, we achieve recall {RT_recall} and precision: {RT_precision}, fix precision: {RT_fix_prec} and fix recall: {RT_fix_rec}"
    )
    return optimal_t


if __name__ == "__main__":
    args = parse_args()
    args.start_time = time.time()
    args.fac_list = np.arange(
        float(args.fac_list.split(",")[0]),
        float(args.fac_list.split(",")[1]),
        float(args.fac_list.split(",")[2]),
    )
    print(args.fac_list)
    print(
        f"Dataset: {args.Fname}, initial_cost: {args.initial_cost}, total_cost: {args.total_cost}, method: {args.method}, assumption: {args.PQA}"
    )

    Path(f"./results/SPRinT_{args.PQA}/").mkdir(parents=True, exist_ok=True)

    if args.method == "NNH":
        args.D_attr = get_data(
            filename=f"data/medical/{args.Fname}/" + args.Fname + ".testfull"
        )
        args.agg = "mean"
        args.subject = "of NNs of q"
        print(f"Prob: {args.Prob}; r: {args.Dist_t}")
        print(f"H1: {args.agg} {args.attr} {args.subject} is {args.H1_op}")

    Proxy_emb, Oracle_emb = load_data(args)

    for seed in range(1, 11):
        args.optimal_cost = None
        # args.rt = 0.01
        print(f"*********************** start seed {seed} ***********************")
        np.random.seed(seed)

        # prepare data embedding, dist to q, and ground truth
        Index = np.random.choice(
            range(len(Oracle_emb)), size=args.num_query, replace=False
        )

        Proxy_dist, Oracle_dist = preprocess_dist(
            Oracle_emb, Proxy_emb, Oracle_emb[[Index[0]]]
        )
        if args.PQA == "PQA":
            Oracle_dist = preprocess_sync(Proxy_dist, norm_scale)

        if args.method == "P-NNH":
            args.true_ans_D = np.where(Oracle_dist <= args.Dist_t)[0]
            args.agg_D = len(args.true_ans_D) / Oracle_dist.shape[0]
            print(f"the GT proportion is {(args.agg_D)}")
        elif args.method == "NNH":
            args.true_ans_D = np.where(Oracle_dist <= args.Dist_t)[0]
            args.l_D, args.agg_D = agg_value(
                args.D_attr, args.true_ans_D, args.attr_id, args.agg
            )

            _, args.agg_D_full = agg_value(
                args.D_attr, range(len(args.D_attr)), args.attr_id, args.agg
            )
            print(
                f"The number of NN in D is {len(args.true_ans_D)} ({len(args.true_ans_D)/Proxy_dist.shape[0]}%), the GT aggregated value of NN is {args.agg_D} and the aggregated value in D is {args.agg_D_full}"
            )

        # get a sample s
        indices = np.random.choice(Oracle_dist.shape[0], args.total_cost, replace=False)
        oracle_dist_S = Oracle_dist[indices]
        proxy_dist_S = Proxy_dist[indices]
        args.true_ans_S = np.where(oracle_dist_S <= args.Dist_t)[0]
        if args.method == "P-NNH":
            args.agg_S = len(args.true_ans_S) / oracle_dist_S.shape[0]
        else:
            args.S_attr = [args.D_attr[i] for i in indices]
            args.l_S, args.agg_S = agg_value(
                args.S_attr, args.true_ans_S, args.attr_id, args.agg
            )
            _, args.agg_S_full = agg_value(
                args.S_attr, range(len(args.S_attr)), args.attr_id, args.agg
            )
            print(
                f"The number of NN in S is {len(args.true_ans_S)} ({len(args.true_ans_S)/proxy_dist_S.shape[0]}%), the aggregation of true NN is {args.agg_S} and the aggregated value of all data in S is {args.agg_S_full}"
            )

        # fixed sample in s to monitor precision target
        if args.PQA == "PQA":
            args.fix_sample = np.random.choice(
                len(oracle_dist_S), size=int(args.initial_cost), replace=False
            )
        elif args.PQA == "PQE":
            assert (
                args.initial_cost > 100
            ), "Initial cost must be greater than the length of fixed sample."
            args.fix_sample = np.random.choice(
                len(oracle_dist_S), size=int(args.initial_cost - 100), replace=False
            )

        available_indices = np.setdiff1d(np.arange(len(oracle_dist_S)), args.fix_sample)
        args.available_oracle_dist_S = oracle_dist_S[available_indices]
        args.available_proxy_dist_S = proxy_dist_S[available_indices]

        # find optimal rt
        args.rt = find_optimal_rt(args, oracle_dist_S, proxy_dist_S, args.initial_cost)

        # if satisfying PQA, all oracle cost goes to fixed sample
        if args.PQA == "PQA":
            args.optimal_cost = args.initial_cost
        elif args.PQA == "PQE":
            # find optimal cost
            find_cost_start = time.time()
            (
                args.optimal_cost,
                find_cost_cost_list,
                find_cost_r_list,
                find_cost_p_list,
                find_cost_fix_r_list,
                find_cost_fix_p_list,
                find_cost_acc_list,
                RT_precision,
                RT_recall,
            ) = find_optimal_cost(
                args,
                oracle_dist_S,
                proxy_dist_S,
                Oracle_dist,
            )
            find_cost_time = round(time.time() - find_cost_start, 4)

        # rerun algo for 30 times for average results
        (
            avg_acc,
            avg_recall,
            avg_precision,
            avg_fix_recall,
            avg_fix_precision,
            avg_agg,
            avg_CI_l,
            avg_CI_h,
            avg_f1,
            avg_fix_f1,
        ) = PQE_better(
            args,
            Oracle_dist,
            oracle_dist_S,
            proxy_dist_S,
            seed,
        )

        # output results
        if args.PQA == "PQA":
            pass
        elif args.PQA == "PQE":
            if args.method == "P-NNH":
                process_results_P_NNH(
                    args,
                    seed,
                    find_cost_cost_list,
                    find_cost_r_list,
                    find_cost_p_list,
                    find_cost_fix_r_list,
                    find_cost_fix_p_list,
                    find_cost_acc_list,
                    find_cost_time,
                    avg_acc,
                    avg_recall,
                    avg_precision,
                )
        process_results_NNH(
            args,
            seed,
            avg_agg,
            avg_CI_l,
            avg_CI_h,
            avg_f1,
            avg_fix_f1,
            avg_acc,
            avg_recall,
            avg_precision,
            avg_fix_recall,
            avg_fix_precision,
        )
