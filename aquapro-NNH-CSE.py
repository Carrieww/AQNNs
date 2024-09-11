from math import floor, ceil, sqrt
from collections import defaultdict

from aquapro_util import (
    load_data,
    preprocess_dist,
    draw_sample_s_m,
    find_best_sm,
    find_K_pt,
    preprocess_ranks,
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
from hyper_parameter import pilot_eps, pilot_size, eps
import numpy as np

# import pickle
from pathlib import Path

import time


@njit
def test_CSC_RT(
    oracle_dist, ranks, t=0.9, prob=0.9, rt=0.9, mode="approx_m1", K=None, pilots=None
):
    true_ans = np.where(oracle_dist <= t)[0]
    D = len(oracle_dist)

    if K is None:
        K = floor(len(true_ans) * (1 - rt)) + 1

    best_s, best_m = find_best_sm(D=D, K=K, mode=mode, prob=prob)

    samples = draw_sample_s_m(D, best_s, best_m)
    samples_true = samples[np.where(oracle_dist[ranks[samples]] <= t)[0]]
    samples_false = samples[np.where(oracle_dist[ranks[samples]] > t)[0]]

    if len(samples_true) == 0:
        k_star = np.random.choice(len(oracle_dist))
    else:
        k_star = samples[np.max(np.where(oracle_dist[ranks[samples]] <= t)[0])]

    if pilots is None:
        union_samples = ranks[samples]
        union_false = ranks[samples_false]
        cost = len(samples)
    else:
        union_samples = array_union(ranks[samples], ranks[pilots])
        pilots_false = pilots[np.where(oracle_dist[ranks[pilots]] > t)[0]]
        union_false = array_union(ranks[samples_false], ranks[pilots_false])
        cost = len(array_union(samples, pilots))

    ans = set_diff(array_union(ranks[: k_star + 1], union_samples), union_false)

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, best_s, best_m, cost, ans


@njit
def test_CSC_PT(oracle_dist, ranks, t=0.9, prob=0.9, pt=0.9, mode="approx_m1", K=None):
    true_ans = np.where(oracle_dist <= t)[0]
    D = len(oracle_dist)

    if K is None:
        K = 0
        true_pos = 0
        for i in range(len(ranks)):
            if oracle_dist[ranks[i]] <= t:
                true_pos += 1
                if true_pos / (i + 1) >= pt:
                    K += 1
                else:
                    break

    if K == 0:
        samples = list()
        counter = 0
        while True:
            sample = draw_sample_s_m(D, min(D, pilot_size * 2**counter), 1)
            sample_true = sample[np.where(oracle_dist[ranks[sample]] <= t)[0]]
            samples.extend(sample)
            counter += 1
            if len(sample_true) > 0:
                break
        samples = np.unique(np.array(samples))
        true_pos = 0
        k_star = 0

        for j in range(len(samples)):
            if oracle_dist[ranks[samples[j]]] <= t:
                true_pos += 1
            if true_pos / (j + 1) >= pt:
                k_star = samples[j]

        hoeff_num = ceil(-np.log(1 - prob) / (2 * pilot_eps**2))
        hoeff_samples = np.random.choice(k_star + 1, size=hoeff_num, replace=True)
        hoeff_true = hoeff_samples[np.where(oracle_dist[ranks[hoeff_samples]] <= t)[0]]
        hoeff_false = hoeff_samples[np.where(oracle_dist[ranks[hoeff_samples]] > t)[0]]
        precis_lb = len(hoeff_true) / hoeff_num - pilot_eps

        samples_true = samples[np.where(oracle_dist[ranks[samples]] <= t)[0]]
        samples_false = samples[np.where(oracle_dist[ranks[samples]] > t)[0]]

        union_samples = array_union(ranks[hoeff_samples], ranks[samples])
        union_false = array_union(ranks[hoeff_false], ranks[samples_false])
        union_true = array_union(ranks[hoeff_true], ranks[samples_true])

        if precis_lb >= pt:
            ans = set_diff(array_union(ranks[: k_star + 1], union_samples), union_false)
        else:
            ans = union_true

        true_pos = len(np.intersect1d(ans, true_ans))
        precision = true_pos / len(ans)
        recall = true_pos / len(true_ans)

        cost = len(array_union(samples, hoeff_samples))

        return (
            precision,
            recall,
            cost,
            1,
            cost,
            ans,
            array_union(samples, hoeff_samples),
            k_star,
        )

    best_s, best_m = find_best_sm(D=D, K=K, mode=mode, prob=prob)

    samples = draw_sample_s_m(D, best_s, best_m)
    samples_true = samples[np.where(oracle_dist[ranks[samples]] <= t)[0]]
    samples_false = samples[np.where(oracle_dist[ranks[samples]] > t)[0]]

    if len(samples_true) == 0:
        k_star = np.random.choice(len(oracle_dist))
    else:
        k_star = samples[np.min(np.where(oracle_dist[ranks[samples]] <= t)[0])]

    ans = set_diff(
        array_union(ranks[: k_star + 1], ranks[samples]), ranks[samples_false]
    )

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)
    cost = len(samples)

    return precision, recall, best_s, best_m, cost, ans, samples, k_star


def test_CSE_RT(oracle_dist, ranks, delta, t=0.9, prob=0.9, rt=0.9, mode="approx_m1"):
    hoeff_num = ceil(-np.log(delta) / (2 * eps**2))
    hoeff_samples = np.random.choice(len(oracle_dist), size=hoeff_num, replace=True)
    hoeff_true = hoeff_samples[np.where(oracle_dist[ranks[hoeff_samples]] <= t)[0]]
    A_lb = max(1, floor((len(hoeff_true) / hoeff_num - eps) * len(oracle_dist)))
    K_rt = floor(A_lb * (1 - rt)) + 1

    precision, recall, _, _, cost, ans = test_CSC_RT(
        oracle_dist,
        ranks,
        t=t,
        prob=prob,
        rt=rt,
        mode=mode,
        K=K_rt,
        pilots=np.unique(hoeff_samples),
    )

    return precision, recall, cost, ans


def test_CSE_PT(oracle_dist, ranks, t=0.9, prob=0.9, pt=0.9, mode="approx_m1"):
    true_ans = np.where(oracle_dist <= t)[0]
    presample = np.unique(
        np.random.choice(len(oracle_dist), size=pilot_size, replace=False)
    )
    presample_true = presample[np.where(oracle_dist[ranks[presample]] <= t)[0]]
    presample_false = presample[np.where(oracle_dist[ranks[presample]] > t)[0]]

    K_pt = 0
    true_pos = 0
    for i in range(len(presample)):
        if oracle_dist[ranks[presample[i]]] <= t:
            true_pos += 1
            if true_pos / (i + 1) >= pt:
                K_pt += 1
            else:
                break
    K_pt = ceil(K_pt * len(oracle_dist) / pilot_size)

    k_hat = 0
    true_pos = 0
    for i in range(len(presample)):
        if oracle_dist[ranks[presample[i]]] <= t:
            true_pos += 1
            if true_pos / (i + 1) >= pt:
                k_hat = presample[i]

    _, _, _, _, _, ans, samples, k_star = test_CSC_PT(
        oracle_dist, ranks, t=t, prob=prob, pt=pt, mode=mode, K=K_pt
    )

    ans = set_diff(array_union(ans, ranks[presample]), ranks[presample_false])
    cost = len(array_union(ranks[presample], ranks[samples]))

    if K_pt > 0:
        k_star = max(k_star, k_hat)
        hoeff_num = ceil(-np.log(1 - prob) / (2 * pilot_eps**2))
        hoeff_samples = np.random.choice(k_star + 1, size=hoeff_num, replace=True)
        hoeff_true = hoeff_samples[np.where(oracle_dist[ranks[hoeff_samples]] <= t)[0]]
        hoeff_false = hoeff_samples[np.where(oracle_dist[ranks[hoeff_samples]] > t)[0]]
        precis_lb = len(hoeff_true) / hoeff_num - pilot_eps

        samples_true = samples[np.where(oracle_dist[ranks[samples]] <= t)[0]]
        samples_false = samples[np.where(oracle_dist[ranks[samples]] > t)[0]]

        union_samples = array_union(
            ranks[hoeff_samples], array_union(ranks[presample], ranks[samples])
        )
        union_true = array_union(
            ranks[hoeff_true], array_union(ranks[presample_true], ranks[samples_true])
        )
        union_false = array_union(
            ranks[hoeff_false],
            array_union(ranks[presample_false], ranks[samples_false]),
        )

        if precis_lb >= pt:
            ans = set_diff(array_union(ranks[: k_star + 1], union_samples), union_false)
        else:
            ans = union_true

        cost = len(
            array_union(
                ranks[hoeff_samples], array_union(ranks[presample], ranks[samples])
            )
        )

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, cost, ans


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
    # Pt = Rt = 0.72
    Prob = 0.95
    Dist_t = 0.85
    H1_op = "greater"
    seed = 4
    mode = "exact"
    print(f"Prob: {Prob}; r: {Dist_t}; seed: {seed}; mode: {mode}")

    num_query = 1
    num_sample = 30
    # sampling_method = "RNS"
    np.random.seed(seed)
    Index = np.random.choice(range(len(Oracle_emb)), size=num_query, replace=False)

    fac_list = np.arange(0.8, 1.25, 0.1)
    fac_list = [round(num, 4) for num in fac_list]

    if Fname == "icd9_eICU":
        # sample_size_list = [8000,8100,8200,8236]
        sample_size_list = [2000]
        # sample_size_list = list(range(500, 4001, 500))
    elif Fname == "icd9_mimic":
        # sample_size_list = [4000,4100,4200,4244]
        sample_size_list = list(range(1000, 4001, 1000))
    res = defaultdict(list)

    # Proxy_dist, _ = preprocess_dist(Oracle, Proxy, Oracle[[Index[0]]])
    # sync_Oracle = preprocess_sync(Proxy_dist, norm_scale)
    # true_ans = np.where(sync_Oracle <= Dist_t)[0]

    Proxy_dist, Oracle_dist = preprocess_dist(
        Oracle_emb, Proxy_emb, Oracle_emb[[Index[0]]]
    )
    Ranks = preprocess_ranks(Proxy_dist)
    true_ans_D = np.where(Oracle_dist <= Dist_t)[0]

    prop_D = len(true_ans_D) / Oracle_dist.shape[0]
    print(f"the GT proportion is {(prop_D)}")

    delta = 1 - sqrt(Prob)
    rt_prob = Prob / (1 - delta)

    for sample_size in sample_size_list:
        print(f"sample size: {sample_size}")
        rt_k_success = defaultdict(list)
        rt_k_precision = defaultdict(list)
        rt_k_recall = defaultdict(list)
        rt_k_acc = defaultdict(list)
        rt_k_rejH0 = defaultdict(list)
        rt_response_time = defaultdict(list)
        rt_prop = defaultdict(list)
        rt_cost = defaultdict(list)

        pt_k_success = defaultdict(list)
        pt_k_recall = defaultdict(list)
        pt_k_precision = defaultdict(list)
        pt_k_acc = defaultdict(list)
        pt_k_rejH0 = defaultdict(list)
        pt_response_time = defaultdict(list)
        pt_prop = defaultdict(list)
        pt_cost = defaultdict(list)

        for Pt in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print(f"Pt=Rt={Pt}")
            Rt = Pt
            for sample_ind in range(num_sample):
                one_sample_start = time.time()

                indices = np.random.choice(
                    Oracle_dist.shape[0], sample_size, replace=False
                )
                oracle_dist_S = Oracle_dist[indices]
                proxy_dist_S = Proxy_dist[indices]
                ranks_S = preprocess_ranks(proxy_dist_S)
                S_size = oracle_dist_S.shape[0]

                true_ans_S = np.where(oracle_dist_S <= Dist_t)[0]
                print(
                    f"prop_S at {sample_ind}-th sample is: ",
                    true_ans_S.shape[0] / S_size,
                )
                time_one_sample = time.time() - one_sample_start

                # CSC-RT
                RT_start = time.time()
                RT_precision, RT_recall, RT_cost, RT_ans = test_CSE_RT(
                    oracle_dist_S,
                    ranks_S,
                    t=Dist_t,
                    delta=delta,
                    prob=rt_prob,
                    rt=Rt,
                    mode=mode,
                )
                RT_end = time.time()

                # CSC-PT
                PT_start = time.time()
                PT_precision, PT_recall, PT_cost, PT_ans = test_CSE_PT(
                    oracle_dist_S, ranks_S, t=Dist_t, prob=Prob, pt=Pt, mode=mode
                )
                PT_end = time.time()

                for fac in fac_list:
                    c_time_GT = (len(true_ans_D) / Oracle_dist.shape[0]) * fac
                    print(f">>> c is {c_time_GT}")
                    _, _, GT = one_proportion_z_test(
                        len(true_ans_D), Oracle_dist.shape[0], c_time_GT, 0.05, H1_op
                    )
                    print(f"the ground truth to reject H0 result is : {GT}")
                    rt_align, rt_reject = HT_acc(
                        "CSE-RT",
                        RT_ans,
                        S_size,
                        H1_op,
                        GT,
                        c_time_GT,
                    )
                    rt_k_acc[fac].append(rt_align)
                    rt_k_rejH0[fac].append(rt_reject)
                    rt_k_success[fac].append(int(RT_recall >= Rt))
                    rt_k_recall[fac].append(RT_recall)
                    rt_k_precision[fac].append(RT_precision)
                    rt_response_time[fac].append(round(RT_end - RT_start, 4))
                    rt_cost[fac].append(RT_cost)
                    rt_prop[fac].append(len(RT_ans) / S_size)

                    pt_align, pt_reject = HT_acc(
                        "CSE-PT",
                        PT_ans,
                        S_size,
                        H1_op,
                        GT,
                        c_time_GT,
                    )
                    pt_k_acc[fac].append(pt_align)
                    pt_k_rejH0[fac].append(pt_reject)
                    pt_k_success[fac].append(int(PT_recall >= Pt))
                    pt_k_recall[fac].append(PT_recall)
                    pt_k_precision[fac].append(PT_precision)
                    pt_response_time[fac].append(round(PT_end - PT_start, 4))
                    pt_cost[fac].append(PT_cost)
                    pt_prop[fac].append(len(PT_ans) / S_size)

            Path(f"./results_NNH/CSE/").mkdir(parents=True, exist_ok=True)
            # print("final result is:")
            for fac in fac_list:
                print("here")
                backup_res = [
                    Pt,
                    fac,
                    sample_size,
                    round(sum(pt_k_recall[fac]) / len(pt_k_recall[fac]), 4),
                    round(sum(pt_k_precision[fac]) / len(pt_k_precision[fac]), 4),
                    round(sum(pt_k_success[fac]) / len(pt_k_success[fac]), 4),
                    round(sum(pt_k_acc[fac]) / len(pt_k_acc[fac]), 4),
                    round(sum(pt_k_rejH0[fac]) / len(pt_k_rejH0[fac]), 4),
                    round(sum(pt_response_time[fac]) / len(pt_response_time[fac]), 4),
                    round(sum(pt_prop[fac]) / len(pt_prop[fac]), 4),
                    round(sum(pt_cost[fac]) / len(pt_cost[fac]), 4),
                    round(sum(rt_k_recall[fac]) / len(rt_k_recall[fac]), 4),
                    round(sum(rt_k_precision[fac]) / len(rt_k_precision[fac]), 4),
                    round(sum(rt_k_success[fac]) / len(rt_k_success[fac]), 4),
                    round(sum(rt_k_acc[fac]) / len(rt_k_acc[fac]), 4),
                    round(sum(rt_k_rejH0[fac]) / len(rt_k_rejH0[fac]), 4),
                    round(sum(rt_response_time[fac]) / len(rt_response_time[fac]), 4),
                    round(sum(rt_prop[fac]) / len(rt_prop[fac]), 4),
                    round(sum(rt_cost[fac]) / len(rt_cost[fac]), 4),
                ]
                with open(
                    f"results_NNH/CSE/"
                    + Fname
                    + "_"
                    + H1_op
                    + f"_query{str(seed)}_0907_30.txt",
                    "a",
                ) as file:
                    results_str = "\t".join(map(str, backup_res)) + "\n"
                    file.write(results_str)
                # results_str = "\t".join(map(str, backup_res)) + "\n"
                # print(results_str)

    end_time = time.time()
    print("execution time is %.2fs" % (end_time - start_time))
