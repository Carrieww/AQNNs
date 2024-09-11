from math import floor, ceil
from scipy.stats import gaussian_kde, entropy, norm
import pickle
import matplotlib.pyplot as plt
from scipy import stats

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

# from RandomWalkSampler import RandomWalkSampler
from numba import njit
from hyper_parameter import norm_scale, std_offset
import numpy as np
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

    precision, recall, _, ans = test_PQA_PT(
        oracle_dist, phi, topk, t=t, prob=prob, pt=pt, pilots=samples
    )

    return precision, recall, _, ans


def test_PQE_RT(oracle_dist, proxy_dist, bd, t=0.9, prob=0.9, rt=0.9):
    imp_p = (1 - proxy_dist + 1e-3) / np.sum(1 - proxy_dist + 1e-3)
    samples = np.random.choice(len(oracle_dist), size=bd, replace=False, p=imp_p)

    est_scale = np.std(oracle_dist[samples] - proxy_dist[samples]) + std_offset

    topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=est_scale, t=t)

    precision, recall, _, ans = test_PQA_RT(
        oracle_dist, phi, topk, t=t, prob=prob, rt=rt, pilots=samples
    )

    return precision, recall, _, ans


def exp_PQA_maximal_CR(
    D_input,
    attr_id,
    sync_oracle_dist,
    proxy_dist,
    c,
    GT,
    pt=0.9,
    rt=0.9,
    t=0.9,
    prob=0.9,
    fname="",
    is_precompile=False,
    sampling_method="RNS",
    s=1000,
    num_sample=1,
    agg="mean",
    H1_op="two-sided",
):
    rt_k_success = []
    rt_k_precis = []
    rt_k_acc = []
    rt_k_rejH0 = []
    rt_response_time = []
    pt_k_success = []
    pt_k_recall = []
    pt_k_acc = []
    pt_response_time = []
    pt_k_rejH0 = []

    query_time_start = time.time()

    for sample_ind in range(num_sample):
        print(f"<{sample_ind}-th sample>")
        if sync_oracle_dist.shape[0] < s:
            raise ValueError("Sample size cannot be larger than the dataset size.")

        if sampling_method == "RNS":
            indices = np.random.choice(len(sync_oracle_dist), s, replace=False)
            sync_oracle_dist_S = sync_oracle_dist[indices]
            true_ans_S = np.where(sync_oracle_dist_S <= t)[0]
            proxy_dist_S = proxy_dist[indices]
            S_input = [D_input[i] for i in indices]
            S_size = sync_oracle_dist_S.shape[0]

            l_S_full, agg_S_full = agg_value(S_input, range(len(S_input)), attr_id, agg)
            # print("start comparing S and D")
            # print(f"agg S is {agg_S_full}")
            # ks_test(l_S_full, l_D_full)

        else:
            raise Exception("Unknown sampling method")

        # PQA - PT - sync
        topk, phi = preprocess_topk_phi(proxy_dist_S, norm_scale=norm_scale, t=t)
        PT_start = time.time()
        _, _, k_star = test_PQA_PT(sync_oracle_dist_S, phi, topk, t=t, prob=prob, pt=pt)

        if k_star == 0:
            pt_k_success.append(1)
            pt_k_recall.append(0)
            pt_response_time.append(round(time.time() - PT_start, 2))
            # pt_k_acc.append(-1)
            # pt_k_rejH0.append(-1)
        else:
            k_hat = ceil(k_star * (1 + 0 / 100))
            ans = topk[:k_hat]

            true_pos = len(np.intersect1d(ans, true_ans_S))
            precision = true_pos / len(ans)
            recall = true_pos / len(true_ans_S)

            pt_k_success.append(int(precision >= pt))
            pt_k_recall.append(recall)
            pt_response_time.append(round(time.time() - PT_start, 2))

            # extract the target attribute in a list l_S
            l_S, agg_S = agg_value(S_input, ans, attr_id, agg)
            # print("start comparing S and NN_S")
            # print(f"agg NN_S is {agg_S}")
            # stat, p_value = ks_test(l_S_full, l_S)

            # print("start comparing NN_S and NN_O")
            # stat, p_value = ks_test(l_S, l_D)

            print(
                f"The number of NN in S is {len(ans)} ({len(ans)/S_size}%), the aggregated value is {agg_S} (c = {c})"
            )
            # use l_S to perform t-test
            align, S_rejectH0 = HT_acc(l_S, c, H1_op, GT=GT, is_D=False)
            pt_k_acc.append(align)
            pt_k_rejH0.append(S_rejectH0)

        # PQA-RT-sync
        RT_start = time.time()
        _, _, k_star = test_PQA_RT(sync_oracle_dist_S, phi, topk, t=t, prob=prob, rt=rt)

        k_hat = int(k_star * (1 + 0 / 100))
        ans = topk[:k_hat]

        true_pos = len(np.intersect1d(ans, true_ans_S))
        precision = true_pos / len(ans)
        if len(true_ans_S) == 0:
            recall = 0
        else:
            recall = true_pos / len(true_ans_S)

        rt_k_success.append(int(recall >= rt))
        rt_k_precis.append(precision)
        rt_response_time.append(round(time.time() - RT_start, 2))

        # extract the target attribute in a list l_S
        l_S, agg_S = agg_value(S_input, ans, attr_id, agg)
        print(
            f"The number of NN in S is {len(ans)} ({len(ans)/S_size}%), the aggregated value is {agg_S} (c = {c})"
        )
        # use l_S to perform t-test
        align, S_rejectH0 = HT_acc(l_S, c, H1_op, GT=GT, is_D=False)
        rt_k_acc.append(align)
        rt_k_rejH0.append(S_rejectH0)

    print("time for a query:", round(time.time() - query_time_start, 2))
    print("================================")

    backup_res = [
        c,
        s,
        round(sum(pt_k_recall) / len(pt_k_recall), 4),
        round(sum(pt_k_success) / len(pt_k_success), 4),
        round(sum(pt_k_acc) / len(pt_k_acc), 4),
        round(sum(pt_k_rejH0) / len(pt_k_rejH0), 4),
        round(sum(pt_response_time) / len(pt_response_time), 4),
        round(sum(rt_k_precis) / len(rt_k_precis), 4),
        round(sum(rt_k_success) / len(rt_k_success), 4),
        round(sum(rt_k_acc) / len(rt_k_acc), 4),
        round(sum(rt_k_rejH0) / len(rt_k_rejH0), 4),
        round(sum(rt_response_time) / len(rt_response_time), 4),
        prob,
    ]
    if not is_precompile:
        Path(f"./results_CNNH/PQE_{H1_op}/").mkdir(parents=True, exist_ok=True)
        with open(
            f"results_CNNH/PQE_{H1_op}/"
            + fname
            + "_"
            + sampling_method
            + "_"
            + H1_op
            + f"_{str(pt)[2:]}_{str(seed)}_0812.txt",
            "a",
        ) as file:
            results_str = "\t".join(map(str, backup_res)) + "\n"
            file.write(results_str)
    else:
        results_str = "\t".join(map(str, backup_res)) + "\n"
        print(results_str)


def ks_test(L1, L2):
    from scipy.stats import ks_2samp

    statistic, p_value = ks_2samp(L1, L2)

    if p_value < 0.05:
        print(
            f"The KS test is significant {statistic, p_value}, we shall reject the null hypothesis. The two distributions are different."
        )
    else:
        print(
            f"The KS test is NOT significant {statistic, p_value}, we shall accept the null hypothesis. The two distributions are similar."
        )

    return statistic, p_value


def HT_acc(l, c, operator, GT=None, is_D=False):
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


def load_attributed_data(Fname):
    # read testref file
    filename_pred = f"/Users/wangyun/Desktop/Graph_HT_2/Dujian/AQUAPRO-main/data/eicu-collaborative-research-database-2/{Fname}.testfull"

    with open(filename_pred, "rb") as file:
        res = pickle.load(file, encoding="latin1")

    return res


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
        if len(l) > 0:
            res = sum(l) / len(l)
        else:
            res = np.nan
    else:
        raise Exception(f"The case for {agg} has not been implemented yet")
    return l, res


def plot(l):
    # Plotting the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(l, bins=10, edgecolor="black", alpha=0.7)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def plot_c_vs_likelihood(op, x, y_rejH0, y_acc):
    plt.figure(figsize=(10, 6))
    marker_list = ["o", "s", "^", "D", "P"]
    default_cycler = plt.rcParams["axes.prop_cycle"]
    colors = [prop["color"] for prop in default_cycler]

    # x_col = 0
    # if query_type == "PT":
    #     y_rejH0 = 5
    #     y_acc = 4
    # elif query_type == "RT":
    #     y_rejH0 = 10
    #     y_acc = 9
    # elif query_type == "combine":
    #     y_rejH0 = 3
    #     y_acc = 2

    # offset = (0.03, 0.05)

    plt.plot(
        x,
        y_acc,
        marker=marker_list[0],
        label="acc",
        color=colors[0],
    )

    # Add text of the likelihood of rejecting H0 to the plot
    for i, e in enumerate(y_acc):
        plt.text(
            x[i],
            e - 0.03,
            f"{y_rejH0[i]:.1f}",
            fontsize=14,
            fontweight="bold",
            color=colors[0],
            backgroundcolor="white",
        )

    # Add labels and title
    plt.xticks(x, fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    # plt.ylim(0, 1.05)
    plt.xlabel("Pt", fontsize=18, fontweight="bold")
    plt.ylabel("HT acc", fontsize=18, fontweight="bold")
    plt.title(
        f"HT acc & likelihood of rejecting H0 vs Pt (PQE, {H1_op}, eICU)",
        fontsize=18,
        fontweight="bold",
    )
    # plt.legend(loc="best", prop={"size": 18, "weight": "bold"})
    plt.tight_layout()
    plt.savefig(
        f"results_CNNH/PQE_{H1_op}/HTacc_"
        + Fname
        + "_"
        + H1_op
        + f"_query{str(seed)}_c{str(c)}_0812.png"
    )


def generate_pdf(l):
    if len(l) == 0:
        return np.nan

    kde = gaussian_kde(l)
    range = np.linspace(min(l), max(l), 1000)
    pdf = kde(range)
    return pdf


if __name__ == "__main__":
    start_time = time.time()
    Fname = "icd9_eICU"
    Proxy_emb, Oracle_emb = load_data(name=Fname)

    # NN algo parameters
    Prob = 0.95
    Dist_t = 0.5
    num_query = 1
    seed = 5
    c = 58
    np.random.seed(seed)
    Index = np.random.choice(range(len(Proxy_emb)), size=num_query, replace=False)

    # hypothesis parameters
    D_attr = get_data(filename="data/eICU_new/" + Fname + ".testfull")
    agg = "mean"
    attr = "age"
    attr_id = 1
    subject = "of NNs of q"
    H1_op = "greater"
    print(f"Prob: {Prob}; r: {Dist_t}; seed: {seed}")
    print(f"H1: {agg} {attr} {subject} is {H1_op} {c}")

    # get NN (percentage of NN) and aggregated value (c) using oracle
    Proxy_dist, _ = preprocess_dist(Oracle_emb, Proxy_emb, Oracle_emb[[Index[0]]])
    sync_Oracle_dist = preprocess_sync(Proxy_dist, norm_scale)
    true_ans_D = np.where(sync_Oracle_dist <= Dist_t)[0]
    l_D, agg_D = agg_value(D_attr, true_ans_D, attr_id, agg)

    # generate density of ages in D
    pdf_D = generate_pdf(l_D)

    _, agg_D_full = agg_value(D_attr, range(len(D_attr)), attr_id, agg)
    print(f"agg_D_full is {agg_D_full}")
    print(
        f"The number of NN in D is {len(true_ans_D)} ({len(true_ans_D)/Proxy_dist.shape[0]}%), the aggregated value is {agg_D}"
    )
    _, D_rejectH0, D_CI_l, D_CI_h = HT_acc(l_D, c, H1_op, is_D=True)
    # print("start comparing D and NN_O")
    # print(f"agg NN_O is {agg_D} and agg D is {agg_D_full}")
    # stat, p_value = ks_test(l_D, l_D_full)

    print(f"The Index target patient is {Index}, with seed = {str(seed)}.")
    rt_k_success = []
    rt_k_precis = []
    rt_k_acc = []
    rt_k_rejH0 = []
    rt_response_time = []
    rt_k_recall = []
    pt_k_success = []
    pt_k_recall = []
    pt_k_acc = []
    pt_response_time = []
    pt_k_precision = []
    pt_k_rejH0 = []
    pt_k_agg = []
    pt_k_CI_l = []
    pt_k_CI_h = []
    rt_k_CI_l = []
    rt_k_CI_h = []
    rt_k_agg = []
    pt_k_kl_divergence = []
    rt_k_kl_divergence = []
    pt_list = np.arange(0.7, 0.09, -0.05)
    pt_list = [round(num, 4) for num in pt_list]

    cost_list = np.arange(3100, 3501, 100)
    # cost_list = [round(num, 4) for num in pt_list]

    for cost in cost_list:
        for Pt in pt_list:
            Rt = Pt
            print(f"\nPt = Rt = {Rt}")

            # PT
            PT_start = time.time()
            PT_precision, PT_recall, PT_k_star, PT_ans = test_PQE_PT(
                sync_Oracle_dist, Proxy_dist, bd=cost, t=Dist_t, prob=Prob, pt=Pt
            )
            PT_end = time.time()
            pt_k_success.append(int(PT_precision >= Pt))
            # print(PT_precision, Pt)
            pt_k_precision.append(PT_precision)
            pt_k_recall.append(PT_recall)
            pt_response_time.append(round(PT_end - PT_start, 4))

            # extract the target attribute in a list l_S
            l_D_PT, agg_PT = agg_value(D_attr, PT_ans, attr_id, agg)

            pdf_PT = generate_pdf(l_D_PT)
            if pdf_PT is not np.nan:
                kl_divergence_PT = entropy(pdf_D, pdf_PT)
            else:
                kl_divergence_PT = np.nan
            print(f"KL Divergence (PT): {kl_divergence_PT}")

            print(
                f"The number of NN in D by PT is {len(PT_ans)} ({len(PT_ans)/Proxy_dist.shape[0]}%), the aggregated value is {agg_PT}"
            )
            align, D_PT_rejectH0, PT_CI_l, PT_CI_h = HT_acc(
                l_D_PT, c, H1_op, GT=D_rejectH0, is_D=False
            )
            pt_k_kl_divergence.append(kl_divergence_PT)
            pt_k_agg.append(agg_PT)
            pt_k_acc.append(align)
            pt_k_rejH0.append(D_PT_rejectH0)
            pt_k_CI_l.append(PT_CI_l)
            pt_k_CI_h.append(PT_CI_h)

            # RT
            RT_start = time.time()
            RT_precision, RT_recall, RT_k_star, RT_ans = test_PQE_RT(
                sync_Oracle_dist, Proxy_dist, bd=cost, t=Dist_t, prob=Prob, rt=Rt
            )
            RT_end = time.time()

            rt_k_success.append(int(RT_recall >= Rt))
            rt_k_recall.append(RT_recall)
            rt_k_precis.append(RT_precision)
            rt_response_time.append(round(RT_end - RT_start, 4))

            # extract the target attribute in a list l_S
            l_D_RT, agg_RT = agg_value(D_attr, RT_ans, attr_id, agg)

            pdf_RT = generate_pdf(l_D_RT)
            if pdf_RT is not np.nan:
                kl_divergence_RT = entropy(pdf_D, pdf_RT)
            else:
                kl_divergence_RT = np.nan
            print(f"KL Divergence (RT): {kl_divergence_RT}")

            print(
                f"The number of NN in S is {len(RT_ans)} ({len(RT_ans)/Proxy_dist.shape[0]}%), the aggregated value is {agg_RT} (c = {c})"
            )
            # use l_S to perform t-test
            align, D_RT_rejectH0, RT_CI_l, RT_CI_h = HT_acc(
                l_D_RT, c, H1_op, GT=D_rejectH0, is_D=False
            )
            rt_k_kl_divergence.append(kl_divergence_RT)
            rt_k_agg.append(agg_RT)
            rt_k_acc.append(align)
            rt_k_rejH0.append(D_RT_rejectH0)
            rt_k_CI_l.append(RT_CI_l)
            rt_k_CI_h.append(RT_CI_h)

            backup_res = [
                Pt,
                round(pt_k_recall[len(pt_k_recall) - 1], 4),
                round(pt_k_precision[len(pt_k_precision) - 1], 4),
                round(pt_k_success[len(pt_k_success) - 1], 4),
                round(pt_k_acc[len(pt_k_acc) - 1], 4),
                round(pt_k_rejH0[len(pt_k_rejH0) - 1], 4),
                round(pt_response_time[len(pt_response_time) - 1], 4),
                round(pt_k_agg[len(pt_k_agg) - 1], 4),
                round(pt_k_CI_l[len(pt_k_CI_l) - 1], 4),
                round(pt_k_CI_h[len(pt_k_CI_h) - 1], 4),
                round(pt_k_kl_divergence[len(pt_k_kl_divergence) - 1], 4),
                round(rt_k_precis[len(rt_k_precis) - 1], 4),
                round(rt_k_recall[len(rt_k_recall) - 1], 4),
                round(rt_k_success[len(rt_k_success) - 1], 4),
                round(rt_k_acc[len(rt_k_acc) - 1], 4),
                round(rt_k_rejH0[len(rt_k_rejH0) - 1], 4),
                round(rt_response_time[len(rt_response_time) - 1], 4),
                round(rt_k_agg[len(rt_k_agg) - 1], 4),
                round(rt_k_CI_l[len(rt_k_CI_l) - 1], 4),
                round(rt_k_CI_h[len(rt_k_CI_h) - 1], 4),
                round(rt_k_kl_divergence[len(rt_k_kl_divergence) - 1], 4),
                cost,
            ]

            Path(f"./results_CNNH/PQE_{H1_op}/").mkdir(parents=True, exist_ok=True)
            with open(
                f"results_CNNH/PQE_{H1_op}/"
                + Fname
                + "_"
                + H1_op
                + f"_query{str(seed)}_c{str(c)}_0812.txt",
                "a",
            ) as file:
                results_str = "\t".join(map(str, backup_res)) + "\n"
                file.write(results_str)

    # plot_c_vs_likelihood(
    #     H1_op,
    #     pt_list,
    #     pt_k_rejH0,
    #     pt_k_acc,
    # )
    print("finish all!")

    # plt.figure(figsize=(10, 8))
    # plt.scatter(pt_list, pt_k_agg, marker="o", label="agg", color="blue")
    # plt.scatter(pt_list, pt_k_CI_l, marker="*", label="Lower CI", color="orange")
    # plt.scatter(pt_list, pt_k_CI_h, marker=".", label="Upper CI", color="black")
    # plt.axhline(
    #     y=agg_D,
    #     linestyle="--",
    #     color="red",
    #     label="GT",
    # )
    # plt.axhline(
    #     y=agg_D_full,
    #     linestyle="--",
    #     color="green",
    #     label="D agg",
    # )
    # plt.axhline(
    #     y=c,
    #     linestyle="--",
    #     color="black",
    #     label="c",
    # )
    # plt.xlabel("Pt", fontsize=18, fontweight="bold")
    # plt.ylabel("Aggregated value", fontsize=18, fontweight="bold")
    # plt.legend(loc="best", prop={"size": 18, "weight": "bold"})
    # plt.savefig(
    #     f"results_CNNH/PQE_{H1_op}/agg_"
    #     + Fname
    #     + "_"
    #     + H1_op
    #     + f"_query{str(seed)}_c{str(c)}_0812.png"
    # )
