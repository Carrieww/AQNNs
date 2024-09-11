from math import floor, ceil, sqrt
from scipy.stats import gaussian_kde, entropy, norm
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import statistics

from aquapro_util import (
    draw_sample_s_m,
    load_data,
    preprocess_dist,
    find_best_sm,
    preprocess_ranks,
    find_K_pt,
)
from aquapro_util import (
    array_union,
    set_diff,
)

# from RandomWalkSampler import RandomWalkSampler
from numba import njit
from hyper_parameter import pilot_size, pilot_eps, eps
import numpy as np
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
        f"HT acc & likelihood of rejecting H0 vs Pt (CSE, {H1_op}, eICU)",
        fontsize=18,
        fontweight="bold",
    )
    # plt.legend(loc="best", prop={"size": 18, "weight": "bold"})
    plt.tight_layout()
    plt.savefig(
        f"results_CNNH/CSE_{H1_op}/HTacc_"
        + Fname
        + "_"
        + H1_op
        + f"_query{str(seed)}_c{str(c)}_0808_{repeat_times}.png"
    )


def generate_pdf(l):
    if len(l) == 0:
        print("l is nan!!!!!")
        return np.nan

    kde = gaussian_kde(l)
    range = np.linspace(min(l), max(l), 1000)
    pdf = kde(range)
    return pdf


def list_mean(lst):
    # remove nan
    print(f"before cleaning, {len(lst)}")
    clean_lst = [x for x in lst if not (isinstance(x, float) and x != x)]
    print(f"after cleaning, {len(clean_lst)}")
    # compute average
    return statistics.mean(clean_lst)


if __name__ == "__main__":
    start_time = time.time()
    Fname = "icd9_eICU"
    Proxy_emb, Oracle_emb = load_data(name=Fname)

    # NN algo parameters
    Prob = 0.95
    Dist_t = 0.5
    num_query = 1
    seed = 5
    c = 66
    repeat_times = 30
    mode = "exact"
    np.random.seed(seed)
    Index = np.random.choice(range(len(Proxy_emb)), size=num_query, replace=False)

    # hypothesis parameters
    D_attr = get_data(filename="data/eICU_new/" + Fname + ".testfull")
    agg = "mean"
    attr = "age"
    attr_id = 1
    subject = "of NNs of q"
    H1_op = "greater"
    print(f"Prob: {Prob}; r: {Dist_t}; seed: {seed}; mode: {mode}")
    print(f"H1: {agg} {attr} {subject} is {H1_op} {c}")

    # get NN (percentage of NN) and aggregated value (c) using oracle
    Proxy_dist, Oracle_dist = preprocess_dist(
        Oracle_emb, Proxy_emb, Oracle_emb[[Index[0]]]
    )
    Ranks = preprocess_ranks(Proxy_dist)
    true_ans_D = np.where(Oracle_dist <= Dist_t)[0]

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

    pt_list = np.arange(0.9, 0.09, -0.05)
    pt_list = [round(num, 4) for num in pt_list]

    delta = 1 - sqrt(Prob)
    rt_prob = Prob / (1 - delta)

    for Pt in pt_list:
        rt_k_success = []
        rt_k_precis = []
        rt_k_acc = []
        rt_k_rejH0 = []
        rt_response_time = []
        rt_k_recall = []
        rt_k_CI_l = []
        rt_k_CI_h = []
        rt_k_agg = []
        rt_k_kl_divergence = []
        rt_cost = []

        pt_k_success = []
        pt_k_recall = []
        pt_k_acc = []
        pt_response_time = []
        pt_k_precision = []
        pt_k_rejH0 = []
        pt_k_agg = []
        pt_k_CI_l = []
        pt_k_CI_h = []
        pt_k_kl_divergence = []
        pt_cost = []

        print(f"\n>>> t is {Pt}")
        Rt = Pt

        # K_rt = floor(len(true_ans_D) * (1 - Rt)) + 1
        # K_pt = find_K_pt(Ranks, Oracle_dist, t=Dist_t, pt=Pt)
        for range_ind in range(repeat_times):
            print(f"repeat: {range_ind}")
            # CSC-RT
            # K_scaled = K_rt
            RT_start = time.time()
            # RT_precision, RT_recall, _, _, RT_cost, RT_ans = test_CSC_RT(
            #     Oracle_dist, Ranks, t=Dist_t, prob=Prob, rt=Rt, mode="exact", K=K_scaled
            # )
            RT_precision, RT_recall, RT_cost, RT_ans = test_CSE_RT(
                Oracle_dist,
                Ranks,
                t=Dist_t,
                delta=delta,
                prob=rt_prob,
                rt=Rt,
                mode=mode,
            )
            RT_end = time.time()

            rt_k_success.append(int(RT_recall >= Rt))
            rt_k_recall.append(RT_recall)
            rt_k_precis.append(RT_precision)
            rt_response_time.append(round(RT_end - RT_start, 4))
            rt_cost.append(RT_cost)

            # extract the target attribute in a list l_S
            l_D_RT, agg_RT = agg_value(D_attr, RT_ans, attr_id, agg)

            # pdf_RT = generate_pdf(l_D_RT)
            # if pdf_RT is not np.nan:
            #     kl_divergence_RT = entropy(pdf_D, pdf_RT)
            # else:
            #     kl_divergence_RT = np.nan
            # print(f"KL Divergence (RT): {kl_divergence_RT}")

            print(
                f"The number of NN in S is {len(RT_ans)} ({len(RT_ans)/Proxy_dist.shape[0]}%), the aggregated value is {agg_RT} (c = {c})"
            )
            # use l_S to perform t-test
            align, D_RT_rejectH0, RT_CI_l, RT_CI_h = HT_acc(
                l_D_RT, c, H1_op, GT=D_rejectH0, is_D=False
            )
            # rt_k_kl_divergence.append(kl_divergence_RT)
            rt_k_agg.append(agg_RT)
            rt_k_acc.append(align)
            rt_k_rejH0.append(D_RT_rejectH0)
            rt_k_CI_l.append(RT_CI_l)
            rt_k_CI_h.append(RT_CI_h)

            # CSC-PT
            # K_scaled = K_pt
            PT_start = time.time()
            PT_precision, PT_recall, PT_cost, PT_ans = test_CSE_PT(
                Oracle_dist, Ranks, t=Dist_t, prob=Prob, pt=Pt, mode=mode
            )
            PT_end = time.time()

            pt_k_success.append(int(PT_precision >= Pt))
            pt_k_precision.append(PT_precision)
            pt_k_recall.append(PT_recall)
            pt_response_time.append(round(PT_end - PT_start, 4))
            pt_cost.append(PT_cost)

            # extract the target attribute in a list l_S
            l_D_PT, agg_PT = agg_value(D_attr, PT_ans, attr_id, agg)

            # pdf_PT = generate_pdf(l_D_PT)
            # if pdf_PT is not np.nan:
            #     kl_divergence_PT = entropy(pdf_D, pdf_PT)
            # else:
            #     kl_divergence_PT = np.nan
            # print(f"KL Divergence (PT): {kl_divergence_PT}")

            print(
                f"The number of NN in D by PT is {len(PT_ans)} ({len(PT_ans)/Proxy_dist.shape[0]}%), the aggregated value is {agg_PT}"
            )
            align, D_PT_rejectH0, PT_CI_l, PT_CI_h = HT_acc(
                l_D_PT, c, H1_op, GT=D_rejectH0, is_D=False
            )
            # pt_k_kl_divergence.append(kl_divergence_PT)
            pt_k_agg.append(agg_PT)
            pt_k_acc.append(align)
            pt_k_rejH0.append(D_PT_rejectH0)
            pt_k_CI_l.append(PT_CI_l)
            pt_k_CI_h.append(PT_CI_h)

        backup_res = [
            Pt,
            round(sum(pt_k_recall) / len(pt_k_recall), 4),
            round(sum(pt_k_precision) / len(pt_k_precision), 4),
            round(sum(pt_k_success) / len(pt_k_success), 4),
            round(sum(pt_k_acc) / len(pt_k_acc), 4),
            round(sum(pt_k_rejH0) / len(pt_k_rejH0), 4),
            round(sum(pt_response_time) / len(pt_response_time), 4),
            round(sum(pt_k_agg) / len(pt_k_agg), 4),
            round(list_mean(pt_k_CI_l), 4),
            round(list_mean(pt_k_CI_h), 4),
            0,
            # round(sum(pt_k_kl_divergence) / len(pt_k_kl_divergence), 4),
            round(sum(pt_cost) / len(pt_cost), 4),
            round(sum(rt_k_precis) / len(rt_k_precis), 4),
            round(sum(rt_k_recall) / len(rt_k_recall), 4),
            round(sum(rt_k_success) / len(rt_k_success), 4),
            round(sum(rt_k_acc) / len(rt_k_acc), 4),
            round(sum(rt_k_rejH0) / len(rt_k_rejH0), 4),
            round(sum(rt_response_time) / len(rt_response_time), 4),
            round(sum(rt_k_agg) / len(rt_k_agg), 4),
            round(list_mean(rt_k_CI_l), 4),
            round(list_mean(rt_k_CI_h), 4),
            0,
            # round(sum(rt_k_kl_divergence) / len(rt_k_kl_divergence), 4),
            round(sum(rt_cost) / len(rt_cost), 4),
        ]
        # print(backup_res)
        Path(f"./results_CNNH/CSE_{H1_op}/").mkdir(parents=True, exist_ok=True)
        with open(
            f"results_CNNH/CSE_{H1_op}/"
            + Fname
            + "_"
            + H1_op
            + f"_query{str(seed)}_c{str(c)}_0808_{repeat_times}.txt",
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
    #     f"results_CNNH/CSE_{H1_op}/agg_"
    #     + Fname
    #     + "_"
    #     + H1_op
    #     + f"_query{str(seed)}_c{str(c)}_0808_30.png"
    # )
