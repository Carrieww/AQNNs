import time
import numpy as np
from numba import njit
from pathlib import Path
from scipy.stats import norm
from math import floor, ceil
from collections import defaultdict
from hyper_parameter import norm_scale, std_offset

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
    Prob = 0.95
    Dist_t = 0.85
    H1_op = "less"
    seed_l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for seed in seed_l:
        print(f"Prob: {Prob}; r: {Dist_t}; seed: {seed}")

        num_query = 1
        np.random.seed(seed)
        Index = np.random.choice(range(len(Oracle)), size=num_query, replace=False)

        fac_list = np.arange(0.5, 1.51, 0.05)
        fac_list = [round(num, 4) for num in fac_list]

        # res = defaultdict(list)

        Proxy_dist, _ = preprocess_dist(Oracle, Proxy, Oracle[[Index[0]]])
        sync_Oracle = preprocess_sync(Proxy_dist, norm_scale)
        true_ans_D = np.where(sync_Oracle <= Dist_t)[0]
        print(f"the GT proportion is {(len(true_ans_D) / Oracle.shape[0])}")

        # PQA-RT-sync
        RT_start = time.time()
        topk, phi = preprocess_topk_phi(Proxy_dist, norm_scale=norm_scale, t=Dist_t)
        oracle_call_set = set()
        epsilon = 0.01
        itr = 0
        rt_max = 0.8
        rt_min = 0
        while itr < 100:
            print(f">>> iteration for finding t*: {itr}")
            rt = (rt_max + rt_min) / 2
            print(f"rt is {rt}")

            _, _, k_star = test_PQA_RT(
                sync_Oracle, phi, topk, t=Dist_t, prob=Prob, rt=rt
            )
            k_hat = int(k_star * (1 + 0 / 100))
            RT_ans = topk[:k_hat]
            oracle_call_set.update(RT_ans)
            true_pos = len(np.intersect1d(RT_ans, true_ans_D))

            precision = true_pos / len(RT_ans)
            print(f"precision is {precision}")
            if precision > rt:
                rt_min = rt
            else:
                rt_max = rt

            if abs(precision - rt) < epsilon:
                break
            else:
                itr += 1
        recall = true_pos / len(true_ans_D)
        response_time = round(time.time() - RT_start, 2)

        rt_k_precision = []
        rt_k_recall = []
        rt_k_acc = []
        rt_k_rejH0 = []
        rt_response_time = []
        rt_prop = []
        rt_cost = []
        for fac in fac_list:
            c_time_GT = (len(true_ans_D) / sync_Oracle.shape[0]) * fac
            print(f"H1: % NN w.r.t {Index} is {H1_op} {c_time_GT}")
            _, _, GT = one_proportion_z_test(
                len(true_ans_D), sync_Oracle.shape[0], c_time_GT, 0.05, H1_op
            )
            print(f"the ground truth to reject H0 result is : {GT}")
            RT_align, RT_reject = HT_acc(
                "PQA-RT",
                RT_ans,
                sync_Oracle.shape[0],
                H1_op,
                GT,
                prop_default=c_time_GT,
            )
            rt_k_acc.append(RT_align)
            rt_k_rejH0.append(RT_reject)
            rt_k_precision.append(precision)
            rt_k_recall.append(recall)
            rt_cost.append(len(oracle_call_set))
            rt_response_time.append(response_time)
            rt_prop.append(len(RT_ans) / sync_Oracle.shape[0])

        print("================================")

        Path(f"./results_NNH/PQA-better1/").mkdir(parents=True, exist_ok=True)
        backup_res = [
            seed,
            round(rt, 4),
            round(np.mean(rt_cost), 4),
            round(np.mean(rt_k_recall), 4),
            round(np.mean(rt_k_precision), 4),
            round(np.mean(rt_k_acc), 4),
            round(np.mean(rt_response_time), 4),
            round(np.mean(rt_prop), 4),
        ]
        with open(
            f"results_NNH/PQA-better1/" + Fname + "_" + H1_op + f"_1008_version2.txt",
            "a",
        ) as file:
            if seed == seed_l[0]:
                file.write(
                    "seed\trt\toptimal cost\tavg recall\tavg precision\tavg acc\tavg time\tapproximated prop\n"
                )
            results_str = "\t".join(map(str, backup_res)) + "\n"
            file.write(results_str)

        # results_str = "\t".join(map(str, backup_res)) + "\n"
        # print(results_str)

    end_time = time.time()
    print("execution time is %.2fs" % (end_time - start_time))
