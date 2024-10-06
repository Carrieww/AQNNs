from math import floor, ceil
from collections import defaultdict

from aquapro_util import (
    load_data,
    preprocess_dist,
    preprocess_topk_phi,
    preprocess_sync,
)
from aquapro_util import (
    array_union,
    set_diff,
)

# from aquapro_util import (
#     baseline_topk_phi_i,
#     baseline_topk_pi,
#     baseline_topk_topc_tableu,
#     baseline_topk_xf,
# )

from numba import njit
from hyper_parameter import norm_scale, std_offset
import numpy as np

# import pickle
from pathlib import Path

import time


@njit
def test_PQA_PT(oracle_dist, phi, topk, t=0.9, prob=0.9, pt=0.9, pilots=None):
    true_ans = np.where(oracle_dist <= t)[0]
    if len(true_ans) == 0:
        return 0, 0, 0

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

        if precis_prob >= prob:
            k_star = i

    if k_star == 0:
        return 1, 0, 0

    if pilots is None:
        ans = topk[:k_star]
    else:
        pilots_false = pilots[np.where(oracle_dist[pilots] > t)[0]]
        ans = set_diff(array_union(topk[:k_star], pilots), pilots_false)

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, k_star


@njit
def test_PQA_RT(oracle_dist, phi, topk, t=0.9, prob=0.9, rt=0.9, pilots=None):
    true_ans = np.where(oracle_dist <= t)[0]
    if len(true_ans) == 0:
        return 0, 0, len(oracle_dist)

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

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, k_star


def test_PQE_PT(oracle_dist, proxy_dist, bd, t=0.9, prob=0.9, pt=0.9):
    imp_p = (1 - proxy_dist + 1e-3) / np.sum(1 - proxy_dist + 1e-3)
    samples = np.random.choice(len(oracle_dist), size=bd, replace=False, p=imp_p)

    est_scale = np.std(oracle_dist[samples] - proxy_dist[samples]) + std_offset

    topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=est_scale, t=t)

    precision, recall, _ = test_PQA_PT(
        oracle_dist, phi, topk, t=t, prob=prob, pt=pt, pilots=samples
    )

    return precision, recall, _


def test_PQE_RT(oracle_dist, proxy_dist, bd, t=0.9, prob=0.9, rt=0.9):
    imp_p = (1 - proxy_dist + 1e-3) / np.sum(1 - proxy_dist + 1e-3)
    samples = np.random.choice(len(oracle_dist), size=bd, replace=False, p=imp_p)

    est_scale = np.std(oracle_dist[samples] - proxy_dist[samples]) + std_offset

    topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=est_scale, t=t)

    precision, recall, _ = test_PQA_RT(
        oracle_dist, phi, topk, t=t, prob=prob, rt=rt, pilots=samples
    )

    return precision, recall, _


def exp_PQA_maximal_CR(
    sync_oracle,
    proxy_dist,
    true_ans,
    pt=0.9,
    rt=0.9,
    t=0.9,
    prob=0.9,
    fname="",
    op="two-sided",
    factor=0.9,
    is_precompile=False,
):
    rt_k_success = defaultdict(list)
    rt_k_precision = defaultdict(list)
    rt_k_recall = defaultdict(list)
    rt_k_acc = defaultdict(list)
    rt_k_rejH0 = defaultdict(list)
    rt_response_time = defaultdict(list)
    rt_prop = defaultdict(list)
    pt_k_success = defaultdict(list)
    pt_k_recall = defaultdict(list)
    pt_k_precision = defaultdict(list)
    pt_k_acc = defaultdict(list)
    pt_response_time = defaultdict(list)
    pt_k_rejH0 = defaultdict(list)
    pt_prop = defaultdict(list)

    scale_list = np.array([0])

    c_time_GT = (len(true_ans) / sync_oracle.shape[0]) * factor
    print(f"c is {c_time_GT}")
    _, _, GT = one_proportion_z_test(
        len(true_ans), sync_oracle.shape[0], c_time_GT, 0.05, op
    )
    print(f"the ground truth to reject H0 result is : {GT}")

    # PQA-PT-sync
    topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=norm_scale, t=t)
    PT_start = time.time()
    _, _, k_star = test_PQA_PT(sync_oracle, phi, topk, t=t, prob=prob, pt=pt)

    for j in scale_list:
        if k_star == 0:
            pt_k_success[j].append(0)
            pt_k_recall[j].append(0)
            pt_k_precision[j].append(0)

            align, reject = HT_acc(
                "PQA-PT",
                [],
                sync_oracle.shape[0],
                op,
                GT,
                prop_default=c_time_GT,
            )
            pt_k_acc[j].append(align)
            pt_k_rejH0[j].append(reject)
        else:
            k_hat = ceil(k_star * (1 + j / 100))
            PT_ans = topk[:k_hat]

            true_pos = len(np.intersect1d(PT_ans, true_ans))
            precision = true_pos / len(PT_ans)
            recall = true_pos / len(true_ans)

            pt_k_success[j].append(int(precision >= pt))
            pt_k_recall[j].append(recall)
            pt_k_precision[j].append(precision)
            pt_response_time[j].append(round(time.time() - PT_start, 2))
            pt_prop[j].append(len(PT_ans) / sync_oracle.shape[0])

            align, reject = HT_acc(
                "PQA-PT",
                PT_ans,
                sync_oracle.shape[0],
                op,
                GT,
                prop_default=c_time_GT,
            )

            pt_k_acc[j].append(align)
            pt_k_rejH0[j].append(reject)

    # PQA-RT-sync
    RT_start = time.time()
    _, _, k_star = test_PQA_RT(sync_oracle, phi, topk, t=t, prob=prob, rt=rt)

    for j in scale_list:
        k_hat = int(k_star * (1 + j / 100))
        RT_ans = topk[:k_hat]

        true_pos = len(np.intersect1d(RT_ans, true_ans))
        precision = true_pos / len(RT_ans)
        recall = true_pos / len(true_ans)

        rt_k_success[j].append(int(recall >= rt))
        rt_k_recall[j].append(recall)
        rt_k_precision[j].append(precision)
        rt_response_time[j].append(round(time.time() - RT_start, 2))
        rt_prop[j].append(len(RT_ans) / sync_oracle.shape[0])

        align, reject = HT_acc(
            "PQA-RT",
            RT_ans,
            sync_oracle.shape[0],
            op,
            GT,
            prop_default=c_time_GT,
        )
        rt_k_acc[j].append(align)
        rt_k_rejH0[j].append(reject)
    # print("time for a query:", round(time.time() - query_time_start, 2))
    print("================================")

    backup_res = [
        pt,
        factor,
        round(sum(pt_k_recall[0]) / len(pt_k_recall[0]), 4),
        round(sum(pt_k_precision[0]) / len(pt_k_precision[0]), 4),
        round(sum(pt_k_success[0]) / len(pt_k_success[0]), 4),
        round(sum(pt_k_acc[0]) / len(pt_k_acc[0]), 4),
        round(sum(pt_k_rejH0[0]) / len(pt_k_rejH0[0]), 4),
        round(sum(pt_response_time[0]) / len(pt_response_time[0]), 4),
        round(sum(pt_prop[0]) / len(pt_prop[0]), 4),
        rt,
        round(sum(rt_k_recall[0]) / len(rt_k_recall[0]), 4),
        round(sum(rt_k_precision[0]) / len(rt_k_precision[0]), 4),
        round(sum(rt_k_success[0]) / len(rt_k_success[0]), 4),
        round(sum(rt_k_acc[0]) / len(rt_k_acc[0]), 4),
        round(sum(rt_k_rejH0[0]) / len(rt_k_rejH0[0]), 4),
        round(sum(rt_response_time[0]) / len(rt_response_time[0]), 4),
        round(sum(rt_prop[0]) / len(rt_prop[0]), 4),
    ]
    if not is_precompile:
        Path(f"./results_NNH/PQA-only/").mkdir(parents=True, exist_ok=True)
        with open(
            f"results_NNH/PQA-only/"
            + fname
            + "_"
            + op
            + f"_query{str(seed)}_0927_30.txt",
            "a",
        ) as file:
            results_str = "\t".join(map(str, backup_res)) + "\n"
            file.write(results_str)
    else:
        results_str = "\t".join(map(str, backup_res)) + "\n"
        print(results_str)


def HT_acc(name, ans, total, op, GT, prop_default=None):
    print(f"finished {name} algorithm for q")
    print(f"FRNN result: {len(ans)}")
    print(f"total patients: {total}")

    if prop_default:
        prop = prop_default
    else:
        prop = len(ans) / total
    print(f"D proportion or c: {prop}; approx: {len(ans) / total}")
    z_stat, p_value, reject = one_proportion_z_test(len(ans), total, prop, 0.05, op)

    print("Z-Statistic:", z_stat)
    print("P-Value:", p_value)
    print("Reject Null Hypothesis:", reject)

    align = reject == GT
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


# pre_compile()
if __name__ == "__main__":
    start_time = time.time()
    Fname = "icd9_eICU"
    Proxy, Oracle = load_data(name=Fname)

    # NN algo parameters
    # t_star = 0.625
    Pt_list = [0.6, 0.7]
    Prob = 0.95
    Dist_t = 0.85
    H1_op = "less"
    seed = 1
    print(f"Prob: {Prob}; r: {Dist_t}; seed: {seed}; Pt=Rt: {Pt_list}")

    num_query = 1
    # num_sample = 50
    np.random.seed(seed)
    Index = np.random.choice(range(len(Oracle)), size=num_query, replace=False)

    fac_list = np.arange(0.8, 1.25, 0.5)
    fac_list = [round(num, 4) for num in fac_list]

    if Fname == "icd9_eICU":
        # sample_size_list = [8000,8100,8200,8236]
        sample_size_list = [320, 420, 520, 620]
        # sample_size_list = list(range(500, 4001, 250))
    elif Fname == "icd9_mimic":
        # sample_size_list = [4000,4100,4200,4244]
        sample_size_list = list(range(1000, 4001, 1000))
    res = defaultdict(list)

    Proxy_dist, _ = preprocess_dist(Oracle, Proxy, Oracle[[Index[0]]])
    sync_Oracle = preprocess_sync(Proxy_dist, norm_scale)
    true_ans_D = np.where(sync_Oracle <= Dist_t)[0]
    print(f"the GT proportion is {(len(true_ans_D) / Oracle.shape[0])}")

    for Pt in Pt_list:
        Rt = Pt
        print(f"pt = rt = {Pt}")
        for fac in fac_list:
            print(f"H1: % NN w.r.t {Index} is {H1_op} {fac}")
            for sample_size in sample_size_list:
                exp_PQA_maximal_CR(
                    sync_Oracle,
                    Proxy_dist,
                    true_ans_D,
                    pt=Pt,
                    rt=Rt,
                    t=Dist_t,
                    prob=Prob,
                    fname=Fname,
                    op=H1_op,
                    factor=fac,
                    is_precompile=False,
                )

    end_time = time.time()
    print("execution time is %.2fs" % (end_time - start_time))
