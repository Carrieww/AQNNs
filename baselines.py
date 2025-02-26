import numpy as np
import pandas as pd
from numba import njit
from math import floor, ceil
from scipy.stats import norm
from scipy.integrate import quad
from hyper_parameter import std_offset
from supg.supg.selector import ApproxQuery
from supg.supg.experiments.experiment import run_experiment
from aquapro_util import array_union, set_diff, preprocess_topk_phi, verbose_print


def test_PQE(args, oracle_dist, proxy_dist, variant, variant_value):
    samples = np.random.choice(len(oracle_dist), size=args.s_p, replace=False)
    est_scale = np.std(oracle_dist[samples] - proxy_dist[samples]) + std_offset
    topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=est_scale, t=args.Dist_t)

    func_map = {
        "PT": lambda: PQA_PT(
            oracle_dist,
            phi,
            topk,
            t=args.Dist_t,
            prob=args.Prob,
            pt=variant_value,
            pilots=samples,
        ),
        "RT": lambda: PQA_RT(
            oracle_dist,
            phi,
            topk,
            t=args.Dist_t,
            prob=args.Prob,
            rt=variant_value,
            pilots=samples,
        ),
    }

    try:
        precision, recall, ans = func_map[variant]()
    except KeyError:
        raise ValueError("Invalid variant specified: expected 'PT' or 'RT'.")

    verbose_print(args, f"true precision is {precision}, true recall is {recall}")

    return precision, recall, None, ans, None, None


# @njit
def PQA_PT(oracle_dist, phi, topk, t=0.9, prob=0.9, pt=0.9, pilots=None):
    true_ans = np.where(oracle_dist <= t)[0]
    if len(true_ans) == 0:
        return 0, 0, true_ans

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
        return 1, 0, topk[:k_star]

    if pilots is None:
        ans = topk[:k_star]
    else:
        pilots_false = pilots[np.where(oracle_dist[pilots] > t)[0]]
        ans = set_diff(array_union(topk[:k_star], pilots), pilots_false)

    true_pos = len(np.intersect1d(ans, true_ans))
    precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, ans


@njit
def PQA_RT(oracle_dist, phi, topk, t=0.9, prob=0.9, rt=0.9, pilots=None):
    true_ans = np.where(oracle_dist <= t)[0]
    if len(true_ans) == 0:
        return 0, 0, true_ans

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

    return precision, recall, ans


def test_topk(oracle_dist, proxy_dist, scale, t=0.9, prob=0.9):
    true_ans = np.where(oracle_dist <= t)[0]
    k = len(true_ans)
    if k < 2:
        return 0, 0, 0, []
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

    return precision, recall, len(table_c), ans


def baseline_topk_xf(sp, sk, p_d, short_dist, norm_scale):
    def intergrand(x):
        xf = np.prod(1 - norm.cdf(x=x, loc=short_dist, scale=norm_scale))
        return norm.pdf(x, loc=p_d, scale=norm_scale) * xf

    intgrl = quad(intergrand, sp, sk)[0]

    term_3 = norm.cdf(x=sp, loc=p_d, scale=norm_scale) * np.prod(
        1 - norm.cdf(x=sp, loc=short_dist, scale=norm_scale)
    )

    return intgrl + term_3


def baseline_topk_topc_tableu(oracle_dist, table_c, k):
    table_all = np.arange(len(oracle_dist))
    k2v_c = sorted([(_, oracle_dist[int(_)]) for _ in table_c], key=lambda x: x[1])
    topk_c = [int(_[0]) for _ in k2v_c[:k]]
    table_u = np.setdiff1d(table_all, table_c)

    return topk_c, table_u


def baseline_topk_pi(proxy_dist, norm_scale, s):
    f = 1 - norm.cdf(x=s, loc=proxy_dist, scale=norm_scale)

    return np.prod(f)


def baseline_topk_phi_i(proxy_dist, norm_scale, sk, sp):
    f_sk = 1 - norm.cdf(x=sk, loc=proxy_dist, scale=norm_scale)
    f_sp = 1 - norm.cdf(x=sp, loc=proxy_dist, scale=norm_scale)

    f_sp = np.clip(f_sp, a_min=1e-8, a_max=None)

    return (1 - f_sk) / f_sp


def SUPG(oracle_dist, proxy_dist, t, primary_target, p, cost, query_type):
    data = pd.DataFrame({"oracle": oracle_dist, "proxy": proxy_dist})
    data = data.sort_values("proxy", axis=0, ascending=True).reset_index()
    data["oracle"] = (data["oracle"] <= t).astype(int)
    data["proxy"] = 1 - data["proxy"]
    data = data.rename(
        columns={"index": "id", "oracle": "label", "proxy": "proxy_score"}
    )

    if query_type == "RT":
        exp_spec = {
            "source": "outside",
            "sampler": "ImportanceSampler",
            "estimator": "None",
            "query": ApproxQuery(
                qtype="rt",
                min_recall=primary_target,
                min_precision=-1,
                delta=1 - p,
                budget=cost,
            ),
            "selector": "ImportanceRecall",
            "num_trials": 1,
        }
    elif query_type == "PT":
        exp_spec = {
            "source": "outside",
            "sampler": "ImportanceSampler",
            "estimator": "None",
            "query": ApproxQuery(
                qtype="pt",
                min_precision=primary_target,
                min_recall=-1,
                delta=1 - p,
                budget=cost,
            ),
            "selector": "ImportancePrecisionSelector",
            "num_trials": 1,
        }
    else:
        print("unknown query type:", query_type)
        exp_spec = {}

    [precision, recall, prob_s, na_rate], ANS = run_experiment(
        cur_experiment=exp_spec, df=data
    )

    return precision, recall, prob_s, na_rate, ANS
