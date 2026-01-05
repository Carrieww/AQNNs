from math import ceil

import numpy as np
from numba import njit

from util import (
    array_union,
    compute_f1_score,
    preprocess_topk_phi,
    set_diff,
    verbose_print,
)

norm_scale = 0.1
std_offset = 0.01


@njit
def SPRinT(Dist_t, Prob, fix_sample, oracle_dist, phi, topk, pt=0.9, pilots=None):
    """
    Core SPRinT algorithm implementation.

    Args:
        Dist_t: Distance threshold
        Prob: Probability threshold
        fix_sample: Fixed sample indices
        oracle_dist: Oracle distances
        phi: Phi values for probability calculation
        topk: Top-k indices
        pt: Precision threshold
        pilots: Pilot sample indices

    Returns:
        precision, recall, k_star, ans, fix_precision, fix_recall
    """
    true_ans = np.where(oracle_dist <= Dist_t)[0]
    if len(true_ans) == 0:
        return 0, 0, 0, np.empty(0, dtype=np.int64), 0, 0

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

        if precis_prob >= Prob:
            k_star = i

    if k_star == 0:
        return 1, 0, k_star, topk[:k_star], 0, 0

    ans = topk[:k_star]

    # fixed sample's precision
    fix_true_ans = fix_sample[np.where(oracle_dist[fix_sample] <= Dist_t)[0]]
    fix_true_pos = len(np.intersect1d(ans, fix_true_ans))

    if len(np.intersect1d(ans, fix_sample)) == 0:
        fix_precision = 0
    else:
        fix_precision = fix_true_pos / len(np.intersect1d(ans, fix_sample))
    if len(fix_true_ans) == 0:
        fix_recall = 0
    else:
        fix_recall = fix_true_pos / len(fix_true_ans)

    if pilots is None:
        ans = topk[:k_star]
    else:
        pilots_false = pilots[np.where(oracle_dist[pilots] > Dist_t)[0]]
        ans = set_diff(array_union(topk[:k_star], pilots), pilots_false)

    true_pos = len(np.intersect1d(ans, true_ans))
    if len(ans) == 0:
        precision = 0
    else:
        precision = true_pos / len(ans)
    recall = true_pos / len(true_ans)

    return precision, recall, k_star, ans, fix_precision, fix_recall


def test_SPRinT(args, oracle_dist, proxy_dist, pt, find_pt=True):
    """
    Test SPRinT algorithm with given parameters.

    Args:
        args: Arguments object
        oracle_dist: Oracle distances
        proxy_dist: Proxy distances
        pt: Precision threshold
        find_pt: Whether to find optimal precision target

    Returns:
        precision, recall, k_star, ans, fix_prec, fix_rec
    """
    if find_pt:
        args.samples = None
    else:
        args.samples = args.fix_sample

    est_scale = (
        np.std(oracle_dist[args.fix_sample] - proxy_dist[args.fix_sample]) + std_offset
    )
    topk, phi = preprocess_topk_phi(proxy_dist, norm_scale=est_scale, t=args.Dist_t)

    precision, recall, _, ans, fix_prec, fix_rec = SPRinT(
        args.Dist_t,
        args.Prob,
        args.fix_sample,
        oracle_dist,
        phi,
        topk,
        pt=pt,
        pilots=args.samples,
    )

    verbose_print(
        args,
        f"true precision is {precision}, true recall is {recall}, and the precision for fixed sample is {fix_prec}, and recall for fixed sample is {fix_rec}",
    )
    verbose_print(
        args,
        f"true f1 is {compute_f1_score(args, precision, recall)}, and fix f1 is {compute_f1_score(args, fix_prec, fix_rec)}",
    )

    return precision, recall, _, ans, fix_prec, fix_rec


def f1_score(args, t, oracle_dist_S, proxy_dist_S):
    """
    Calculate F1 score for given threshold.

    Args:
        args: Arguments object
        t: Threshold value
        oracle_dist_S: Oracle distances for subset S
        proxy_dist_S: Proxy distances for subset S

    Returns:
        F1 score
    """
    _, _, _, _, fix_prec, fix_rec = test_SPRinT(args, oracle_dist_S, proxy_dist_S, t)
    if fix_prec + fix_rec == 0:
        return 0
    else:
        return compute_f1_score(args, fix_prec, fix_rec)


def find_optimal_pt(args, oracle_dist_S, proxy_dist_S, pt=0.0001):
    """
    Find optimal recall target based on algorithm type.

    Args:
        args: Arguments object
        oracle_dist_S: Oracle distances for subset S
        proxy_dist_S: Proxy distances for subset S
        pt: Initial precision target

    Returns:
        optimal_t, CANNOT flag
    """
    CANNOT = 0

    if args.algo == "SPRinT-C":
        # Cost-based optimization
        max_t = 1
        min_t = pt
        fix_rec = 1
        fix_prec = 0
        while abs(fix_rec - fix_prec) > 0.0001 or (fix_prec == 0 and fix_rec == 0):
            pt = (max_t + min_t) / 2
            verbose_print(args, f"---------- finding pt: {pt} ----------")

            _, _, _, ans, fix_prec, fix_rec = test_SPRinT(
                args, oracle_dist_S, proxy_dist_S, pt
            )

            if fix_prec < fix_rec:
                min_t = pt
            else:
                max_t = pt

            if abs(min_t - max_t) < 0.0001:
                verbose_print(args, "CANNOT find optimal pt!!")
                CANNOT = 1
                break
            verbose_print(
                args,
                f"pt = {pt}: Found NN: {len(ans)}, Pilot Precision {fix_prec}, Pilot Recall {fix_rec}",
            )

        optimal_t = pt
        verbose_print(
            args,
            f"Found optimal recall target={optimal_t} with fix prec {fix_prec} and fix rec {fix_rec}",
        )

    elif args.algo == "SPRinT-V":
        # Value-based optimization using F1 score
        left = 0
        right = 1
        tolerance = 0.01
        while right - left > tolerance:
            m1 = left + (right - left) / 3
            m2 = right - (right - left) / 3
            verbose_print(args, f"---------- finding pt between {m1, m2} ----------")
            f1_m1 = f1_score(args, m1, oracle_dist_S, proxy_dist_S)
            f1_m2 = f1_score(args, m2, oracle_dist_S, proxy_dist_S)

            if f1_m1 <= f1_m2:
                left = m1
            else:
                right = m2
        optimal_t = (left + right) / 2
        max_f1 = f1_score(args, optimal_t, oracle_dist_S, proxy_dist_S)
        verbose_print(
            args,
            f"Found optimal recall target={optimal_t} with fix f1 {max_f1}",
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    return optimal_t, CANNOT


class SPRinTRunner:
    """
    Class to run SPRinT algorithm experiments.
    """

    def __init__(self, args):
        self.args = args

    def run_single_experiment(self, oracle_dist_S, proxy_dist_S):
        """
        Run a single SPRinT experiment.

        Args:
            oracle_dist_S: Oracle distances for subset S
            proxy_dist_S: Proxy distances for subset S

        Returns:
            precision, recall, k_star, ans, fix_prec, fix_rec, CANNOT
        """
        self.args.optimal_cost = self.args.s_p

        # Find optimal precision target
        self.args.pt, CANNOT = find_optimal_pt(self.args, oracle_dist_S, proxy_dist_S)

        # Run SPRinT algorithm
        prec, rec, _, ANS, fix_prec, fix_rec = test_SPRinT(
            self.args, oracle_dist_S, proxy_dist_S, self.args.pt, find_pt=False
        )

        return prec, rec, _, ANS, fix_prec, fix_rec, CANNOT
