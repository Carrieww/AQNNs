from math import floor, ceil, sqrt
from collections import defaultdict
from scipy.stats import norm
from aquapro_util import (
    load_data,
    preprocess_dist,
    preprocess_topk_phi,
    preprocess_sync,
)
from aquapro_util import (
    draw_sample_s_m,
    array_union,
    set_diff,
    find_best_sm,
)
from aquapro_util import (
    baseline_topk_phi_i,
    baseline_topk_pi,
    baseline_topk_topc_tableu,
    baseline_topk_xf,
)
from numba import njit
from hyper_parameter import norm_scale, eps, pilot_size, pilot_eps, std_offset
import numpy as np
import pickle
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

    return precision, recall, best_s, best_m, cost


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

    return precision, recall, cost


def test_CSE_RT(oracle_dist, ranks, delta, t=0.9, prob=0.9, rt=0.9, mode="approx_m1"):
    hoeff_num = ceil(-np.log(delta) / (2 * eps**2))
    hoeff_samples = np.random.choice(len(oracle_dist), size=hoeff_num, replace=True)
    hoeff_true = hoeff_samples[np.where(oracle_dist[ranks[hoeff_samples]] <= t)[0]]
    A_lb = max(1, floor((len(hoeff_true) / hoeff_num - eps) * len(oracle_dist)))
    K_rt = floor(A_lb * (1 - rt)) + 1

    precision, recall, _, _, cost = test_CSC_RT(
        oracle_dist,
        ranks,
        t=t,
        prob=prob,
        rt=rt,
        mode=mode,
        K=K_rt,
        pilots=np.unique(hoeff_samples),
    )

    return precision, recall, cost


def test_sample2test_PT(oracle_dist, ranks, bd, t=0.9, pt=0.9):
    true_ans = np.where(oracle_dist <= t)[0]
    inc_true_pos = 0
    k_star = 0

    sample = sorted(np.random.choice(len(oracle_dist), size=bd, replace=False))

    for j in range(len(sample)):
        if oracle_dist[ranks[sample[j]]] <= t:
            inc_true_pos += 1
        if inc_true_pos / (j + 1) >= pt:
            k_star = sample[j]

    ans = ranks[: k_star + 1]

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, bd


def test_sample2test_RT(oracle_dist, ranks, bd, t=0.9, rt=0.9):
    true_ans = np.where(oracle_dist <= t)[0]
    inc_true_pos = 0
    k_star = 0

    sample = np.random.choice(len(oracle_dist), size=bd, replace=False)
    sample_true_pos = len(sample[np.where(oracle_dist[ranks[sample]] <= t)[0]])
    sample = sorted(sample)

    if sample_true_pos == 0:
        ans = []
    else:
        for j in range(len(sample)):
            if oracle_dist[ranks[sample[j]]] <= t:
                inc_true_pos += 1
            if inc_true_pos / sample_true_pos >= rt:
                k_star = sample[j]
                break
        ans = ranks[: k_star + 1]

    true_pos = len(np.intersect1d(ans, true_ans))
    if len(ans) == 0:
        precision = 1
    else:
        precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, bd


def test_topk(oracle_dist, proxy_dist, scale, t=0.9, prob=0.9):
    true_ans = np.where(oracle_dist <= t)[0]
    k = len(true_ans)
    table_c = np.random.choice(len(oracle_dist), size=k, replace=False)
    topk_c, table_u = baseline_topk_topc_tableu(
        oracle_dist=oracle_dist, table_c=table_c, k=k
    )

    sk = sorted(oracle_dist[topk_c])[-1]
    sp = sorted(oracle_dist[topk_c])[-2]
    pi = baseline_topk_pi(proxy_dist=proxy_dist[table_u], norm_scale=scale, s=sk)

    while pi < prob and len(table_c) < len(oracle_dist):
        phi_all = baseline_topk_phi_i(
            proxy_dist=proxy_dist[table_u], norm_scale=scale, sk=sk, sp=sp
        )
        k2phi = sorted(
            np.stack([table_u, phi_all], axis=-1), key=lambda x: x[1], reverse=True
        )
        gamma = baseline_topk_pi(proxy_dist=proxy_dist[table_u], norm_scale=scale, s=sp)
        max_delta = 0
        max_indx = None
        for idx, idx_phi in k2phi:
            if max_delta > gamma * idx_phi:
                break
            table_u_short = np.setdiff1d(table_u, idx)
            delta = baseline_topk_xf(
                sp=sp,
                sk=sk,
                p_d=proxy_dist[int(idx)],
                short_dist=proxy_dist[table_u_short],
                norm_scale=scale,
            )
            if delta > max_delta:
                max_delta = delta
                max_indx = idx

        if max_indx is not None:
            table_c = np.append(table_c, max_indx)
        else:
            new_sample_size = ceil(len(table_u) / 2)
            new_sample = np.random.choice(table_u, size=new_sample_size, replace=False)
            table_c = np.append(table_c, new_sample)

        topk_c, table_u = baseline_topk_topc_tableu(
            oracle_dist=oracle_dist, table_c=table_c, k=k
        )

        sk = sorted(oracle_dist[topk_c])[-1]
        sp = sorted(oracle_dist[topk_c])[-2]
        pi = baseline_topk_pi(proxy_dist=proxy_dist[table_u], norm_scale=scale, s=sk)

    ans = topk_c

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)
    print(precision, recall, len(table_c))

    return precision, recall, len(table_c)


def exp_PQA_maximal_CR(
    sync_oracle,
    proxy_dist,
    true_ans,
    pt=0.9,
    t=0.9,
    prob=0.9,
):
    pt_k_recall = []
    prop = []

    # PQA-PT-sync
    topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=norm_scale, t=t)
    PT_start = time.time()
    _, _, k_star = test_PQA_PT(sync_oracle, phi, topk, t=t, prob=prob, pt=pt)

    if k_star == 0:
        pt_k_recall.append(0)
    else:
        k_hat = ceil(k_star * (1 + 0 / 100))
        ans = topk[:k_hat]

        true_pos = len(np.intersect1d(ans, true_ans))
        # precision = true_pos / len(ans)
        recall = true_pos / len(true_ans)

        pt_k_recall.append(recall)
        prop.append(len(ans) / sync_oracle.shape[0])

    # print("time for a PT query:", round(time.time() - PT_start, 2))
    # print("================================")
    return pt_k_recall[0], prop[0]
    # PQA-RT-sync
    # RT_start = time.time()
    # _, _, k_star = test_PQA_RT(sync_oracle_S, phi, topk, t=t, prob=prob, rt=rt)
    #
    # for j in scale_list:
    #     k_hat = int(k_star * (1 + j / 100))
    #     ans = topk[:k_hat]
    #
    #     true_pos = len(np.intersect1d(ans, true_ans))
    #     precision = true_pos / len(ans)
    #     recall = true_pos / len(true_ans)
    #
    #     rt_k_success[j].append(int(recall >= rt))
    #     rt_k_precis[j].append(precision)


def HT_acc(name, ans, total, op, GT, prop_default=None):
    print(f"finished {name} algorithm for q")
    print(f"FRNN result: {len(ans)}")
    print(f"total patients: {total}")

    if prop_default:
        prop = prop_default
    else:
        prop = len(ans) / total
    print(f"D proportion: {prop}; approx: {len(ans) / total}")
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


if __name__ == "__main__":
    Prob = 0.95
    Dist_t = 0.3
    seed = 36
    # H1_op = "greater"

    Fname = "icd9_eICU"
    Proxy, Oracle = load_data(name=Fname)

    num_query = 1
    np.random.seed(seed)
    Index = np.random.choice(range(len(Oracle)), size=num_query, replace=False)
    print(f"The Index target patient is {Index}")

    Proxy_dist, _ = preprocess_dist(Oracle, Proxy, Oracle[[Index[0]]])
    sync_Oracle = preprocess_sync(Proxy_dist, norm_scale)
    True_ans = np.where(sync_Oracle <= Dist_t)[0]
    prop = len(True_ans) / Oracle.shape[0]
    print(f"prop is {prop}")

    sampling_method = "RNS"

    pt_list = []
    rt_list = []
    time_list = []
    prop_list = []
    S_trial = 1000
    # S_trial = int(Proxy.shape[0] / 10)
    epsilon_list = [0.005]
    for epsilon in epsilon_list:
        print(f"S_trial is {S_trial}")
        res = defaultdict(list)

        # epsilon = 0.01
        Pt = 0.9
        Rt = 0
        m = 100

        start_time = time.time()
        while abs(Pt - Rt) > epsilon:  # Pt=Rt=t
            rt_l = []
            for i in range(m):
                if sampling_method == "RNS":
                    indices = np.random.choice(len(sync_Oracle), S_trial, replace=False)
                    sync_Oracle_S = sync_Oracle[indices]
                    Proxy_dist_S = Proxy_dist[indices]
                    True_ans_S = np.where(sync_Oracle_S <= Dist_t)[0]
                    S_size = sync_Oracle_S.shape[0]
                    prop_S = len(True_ans_S) / S_size

                else:
                    raise Exception(f"{sampling_method} is not supported.")
                rt, prop_S = exp_PQA_maximal_CR(
                    sync_Oracle_S,
                    Proxy_dist_S,
                    True_ans_S,
                    pt=Pt,
                    t=Dist_t,
                    prob=Prob,
                )
                rt_l.append(rt)
            Rt = sum(rt_l) / len(rt_l)
            Pt = (Rt + Pt) / 2
        print(f"t* is Pt = {Pt}, Rt = {Rt}.")
        time_spent = time.time() - start_time
        print("execution time is %.2fs" % (time_spent))

        time_list.append(round(time_spent, 3))
        prop_list.append(round(prop_S, 3))
        pt_list.append(round(Pt, 3))
        rt_list.append(round(Rt, 3))

    print("sample size t*: " + "\t".join(map(str, epsilon_list)))
    print("pt:" + "\t".join(map(str, pt_list)))
    print("rt:" + "\t".join(map(str, rt_list)))
    print("prop_S:" + "\t".join(map(str, prop_list)))
    print("time:" + "\t".join(map(str, time_list)))
