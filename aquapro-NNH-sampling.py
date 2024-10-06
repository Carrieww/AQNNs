from math import floor, ceil
from collections import defaultdict

from aquapro_util import (
    load_data,
    preprocess_dist,
    preprocess_topk_phi,
    preprocess_ranks,
)
from aquapro_util import (
    array_union,
    set_diff,
)

from numba import njit
from hyper_parameter import std_offset
import numpy as np

# import pickle
from pathlib import Path

import time


@njit
def test_PQA_PT(oracle_dist, phi, topk, t=0.9, prob=0.9, pt=0.9, pilots=None):
    true_ans = np.where(oracle_dist <= t)[0]
    if len(true_ans) == 0:
        return 0, 0, 0, np.empty(0, dtype=np.int64)

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
        return 0, 0, 0, np.empty(0, dtype=np.int64)

    if pilots is None:
        ans = topk[:k_star]
    else:
        pilots_false = pilots[np.where(oracle_dist[pilots] > t)[0]]
        ans = set_diff(array_union(topk[:k_star], pilots), pilots_false)

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, k_star, ans


@njit
def test_PQA_RT(oracle_dist, phi, topk, t=0.9, prob=0.9, rt=0.9, pilots=None):
    true_ans = np.where(oracle_dist <= t)[0]
    if len(true_ans) == 0:
        return 0, 0, len(oracle_dist), np.empty(0, dtype=np.int64)

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

    return precision, recall, k_star, ans


def test_PQE_PT(oracle_dist, proxy_dist, bd, t=0.9, prob=0.9, pt=0.9):
    imp_p = (1 - proxy_dist + 1e-3) / np.sum(1 - proxy_dist + 1e-3)
    samples = np.random.choice(len(oracle_dist), size=bd, replace=False, p=imp_p)

    est_scale = np.std(oracle_dist[samples] - proxy_dist[samples]) + std_offset
    print(f"norm scale is {est_scale}")

    topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=est_scale, t=t)

    # # Step 3: Identify low-confidence points
    # low_confidence_points = np.where(phi < 0.8)[0]

    # # # Step 4: Sample from low-confidence points if enough points exist
    # if len(low_confidence_points) >= bd:
    #     samples = np.random.choice(low_confidence_points, size=bd, replace=False)
    # else:
    #     # If not enough low-confidence points, add some high-confidence points
    #     high_confidence_points = np.where(phi >= 0.8)[0]
    #     additional_samples = np.random.choice(
    #         high_confidence_points, size=bd - len(low_confidence_points), replace=False
    #     )
    #     samples = np.concatenate([low_confidence_points, additional_samples])

    # Step 5: Continue as usual with the precision and recall testing
    precision, recall, _, ans = test_PQA_PT(
        oracle_dist, phi, topk, t=t, prob=prob, pt=pt, pilots=samples
    )

    return precision, recall, _, ans


def test_PQE_RT(oracle_dist, proxy_dist, bd, t=0.9, prob=0.9, rt=0.9):
    imp_p = (1 - proxy_dist + 1e-3) / np.sum(1 - proxy_dist + 1e-3)
    samples = np.random.choice(len(oracle_dist), size=bd, replace=False, p=imp_p)

    est_scale = np.std(oracle_dist[samples] - proxy_dist[samples]) + std_offset

    topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=est_scale, t=t)

    # # Step 3: Identify low-confidence points
    # low_confidence_points = np.where(phi < 0.8)[0]

    # # # Step 4: Sample from low-confidence points if enough points exist
    # if len(low_confidence_points) >= bd:
    #     samples = np.random.choice(low_confidence_points, size=bd, replace=False)
    # else:
    #     # If not enough low-confidence points, add some high-confidence points
    #     high_confidence_points = np.where(phi >= 0.8)[0]
    #     additional_samples = np.random.choice(
    #         high_confidence_points, size=bd - len(low_confidence_points), replace=False
    #     )
    #     samples = np.concatenate([low_confidence_points, additional_samples])

    # Step 5: Continue as usual with the precision and recall testing

    precision, recall, _, ans = test_PQA_RT(
        oracle_dist, phi, topk, t=t, prob=prob, rt=rt, pilots=samples
    )

    return precision, recall, _, ans


def HT_acc(name, ans, total, op, GT, prop_c):
    print(f"finished {name} algorithm for q")
    print(f"FRNN result: {len(ans)}")
    print(f"total patients: {total}")

    print(f"c proportion: {prop_c}; approx: {len(ans) / total}")
    z_stat, p_value, reject = one_proportion_z_test(len(ans), total, prop_c, 0.05, op)

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
    Proxy_emb, Oracle_emb = load_data(name=Fname)

    # NN algo parameters
    Prob = 0.95
    Dist_t = 0.85
    H1_op = "greater"
    seed_l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Prob: {Prob}; r: {Dist_t}; seed list: {seed_l}")

    save_path = f"results_NNH/RS/" + Fname + "_" + H1_op + f"_1006.txt"
    Path(f"./results_NNH/RS/").mkdir(parents=True, exist_ok=True)
    with open(
        save_path,
        "a",
    ) as file:
        file.write("seed\tsample size\tavg prop_S\tavg acc\tavg rejH0\tavg time\n")

    num_query = 1
    num_sample = 30
    fac_list = np.arange(0.5, 1.51, 0.05)
    fac_list = [round(num, 4) for num in fac_list]

    if Fname == "icd9_eICU":
        # sample_size_list = [8000,8100,8200,8236]
        sample_size_list = list(range(200, 2010, 200))
        # sample_size_list = list(range(500, 4001, 500))
    elif Fname == "icd9_mimic":
        # sample_size_list = [4000,4100,4200,4244]
        sample_size_list = list(range(1000, 4001, 1000))
    res = defaultdict(list)

    for seed in seed_l:
        np.random.seed(seed)
        Index = np.random.choice(range(len(Oracle_emb)), size=num_query, replace=False)

        Proxy_dist, Oracle_dist = preprocess_dist(
            Oracle_emb, Proxy_emb, Oracle_emb[[Index[0]]]
        )
        # Ranks = preprocess_ranks(Proxy_dist)
        true_ans_D = np.where(Oracle_dist <= Dist_t)[0]

        prop_D = len(true_ans_D) / Oracle_dist.shape[0]
        print(f"the GT proportion is {(prop_D)}")

        for sample_size in sample_size_list:
            print(f"sample size: {sample_size}")
            acc_l = []
            rejH0_l = []
            time_l = []
            prop_S_l = []

            for sample_ind in range(num_sample):
                one_sample_start = time.time()

                indices = np.random.choice(
                    Oracle_dist.shape[0], sample_size, replace=False
                )
                oracle_dist_S = Oracle_dist[indices]
                proxy_dist_S = Proxy_dist[indices]
                # ranks_S = preprocess_ranks(proxy_dist_S)
                S_size = oracle_dist_S.shape[0]

                true_ans_S = np.where(oracle_dist_S <= Dist_t)[0]
                prop_S = round(true_ans_S.shape[0] / S_size, 4)
                print(
                    f"prop_S at {sample_ind}-th sample is: ",
                    prop_S,
                )
                time_one_sample = time.time() - one_sample_start

                for fac in fac_list:
                    c_time_GT = (len(true_ans_D) / Oracle_dist.shape[0]) * fac
                    print(f">>> c is {c_time_GT}")
                    _, _, GT = one_proportion_z_test(
                        len(true_ans_D),
                        Oracle_dist.shape[0],
                        c_time_GT,
                        0.05,
                        H1_op,
                    )
                    print(f"the ground truth to reject H0 result is : {GT}")
                    align, reject = HT_acc(
                        "RNS",
                        true_ans_S,
                        S_size,
                        H1_op,
                        GT,
                        c_time_GT,
                    )
                    acc_l.append(align)
                    rejH0_l.append(reject)
                    time_l.append(time_one_sample)
                    prop_S_l.append(prop_S)

            backup_res = [
                seed,
                sample_size,
                round(np.mean(prop_S_l), 4),
                round(np.mean(acc_l), 4),
                round(np.mean(rejH0_l), 4),
                round(np.mean(time_l), 4),
            ]
            with open(
                save_path,
                "a",
            ) as file:
                results_str = "\t".join(map(str, backup_res)) + "\n"
                file.write(results_str)

    end_time = time.time()
    print("execution time is %.2fs" % (end_time - start_time))
