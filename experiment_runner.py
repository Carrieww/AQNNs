import time

import numpy as np

from baselines import SUPG, test_PQE, test_topk
from hypothesis_testing import HT_acc_t_test, HT_acc_z_test, one_proportion_z_test
from sprint import SPRinTRunner
from util import agg_value, compute_f1_score, verbose_print

norm_scale = 0.1


class ExperimentRunner:
    """
    Class to run experiments with different algorithms.
    """

    def __init__(self, args):
        self.args = args
        self.sprint_runner = (
            SPRinTRunner(args) if args.algo in ["SPRinT-C", "SPRinT-V"] else None
        )

    def run_algorithm(self, oracle_dist_S, proxy_dist_S, start_sample_time):
        """
        Run the specified algorithm.

        Args:
            oracle_dist_S: Oracle distances for subset S
            proxy_dist_S: Proxy distances for subset S
            start_sample_time: Start time for timing

        Returns:
            precision, recall, optimal_cost, ans, fix_prec, fix_rec, CANNOT
        """
        if self.args.algo in ["SPRinT-C", "SPRinT-V"]:
            return self._run_sprint(oracle_dist_S, proxy_dist_S, start_sample_time)
        elif self.args.algo in ["PQA-RT", "PQA-PT"]:
            return self._run_pqa(oracle_dist_S, proxy_dist_S)
        elif self.args.algo in ["SUPG-RT", "SUPG-PT"]:
            return self._run_supg(oracle_dist_S, proxy_dist_S)
        elif self.args.algo == "TopK":
            return self._run_topk(oracle_dist_S, proxy_dist_S)
        else:
            raise ValueError(f"Unknown algorithm: {self.args.algo}")

    def _run_sprint(self, oracle_dist_S, proxy_dist_S, start_sample_time):
        """Run SPRinT algorithm."""
        before_rt = time.time()
        verbose_print(
            self.args,
            f"Data preparation time: {round(time.time() - start_sample_time, 2)} sec",
        )

        prec, rec, _, ANS, fix_prec, fix_rec, CANNOT = (
            self.sprint_runner.run_single_experiment(oracle_dist_S, proxy_dist_S)
        )

        verbose_print(
            self.args,
            f"Optimal rt search time: {round(time.time() - before_rt, 2)} sec",
        )

        return prec, rec, self.args.s_p, ANS, fix_prec, fix_rec, CANNOT

    def _run_pqa(self, oracle_dist_S, proxy_dist_S):
        """Run PQA algorithm."""
        self.args.optimal_cost = self.args.s_p
        self.args.target = (
            self.args.recall_target
            if self.args.algo == "PQA-RT"
            else self.args.precision_target
        )
        prec, rec, _, ANS, _, _ = test_PQE(
            self.args,
            oracle_dist_S,
            proxy_dist_S,
            self.args.algo[-2:],
            self.args.target,
        )
        return prec, rec, self.args.s_p, ANS, np.nan, np.nan, np.nan

    def _run_supg(self, oracle_dist_S, proxy_dist_S):
        """Run SUPG algorithm."""
        self.args.optimal_cost = self.args.s_p
        self.args.target = (
            self.args.recall_target
            if self.args.algo == "SUPG-RT"
            else self.args.precision_target
        )
        prec, rec, _, _, ANS = SUPG(
            oracle_dist_S,
            proxy_dist_S,
            self.args.Dist_t,
            self.args.target,
            self.args.Prob,
            cost=self.args.s_p,
            query_type=self.args.algo[-2:],
        )
        return prec, rec, self.args.s_p, ANS, np.nan, np.nan, np.nan

    def _run_topk(self, oracle_dist_S, proxy_dist_S):
        """Run TopK algorithm."""

        prec, rec, optimal_cost, ANS = test_topk(
            oracle_dist=oracle_dist_S,
            proxy_dist=proxy_dist_S,
            scale=norm_scale,
            t=self.args.Dist_t,
            prob=self.args.Prob,
        )
        return prec, rec, optimal_cost, ANS, np.nan, np.nan, np.nan

    def compute_aggregation(self, proxy_dist_S, ANS, args_S_attr=None):
        """
        Compute aggregation value based on algorithm results.

        Args:
            proxy_dist_S: Proxy distances for subset S
            ANS: Algorithm results
            args_S_attr: Attributes for subset S

        Returns:
            approx_agg_S, f1, fix_f1
        """
        if self.args.agg == "pct":
            approx_agg_S = round(len(ANS) / proxy_dist_S.shape[0], 4)
            return approx_agg_S, None, None
        elif self.args.agg == "count":
            approx_agg_S = len(ANS)
            return approx_agg_S, None, None
        else:
            L_S, approx_agg_S = agg_value(
                args_S_attr, ANS, self.args.attr_id, self.args.agg
            )
            return approx_agg_S, None, None

    def run_hypothesis_testing(
        self, Oracle_dist, oracle_dist_S, ANS, L_S=None, f1=None
    ):
        """
        Run hypothesis testing if applicable.

        Args:
            Oracle_dist: Full oracle distances
            oracle_dist_S: Oracle distances for subset S
            ANS: Algorithm results
            L_S: Attributes for subset S
            f1: F1 score

        Returns:
            acc_l, CI_l, f1_l, fix_f1_l
        """
        acc_l = []
        CI_l = []
        f1_l = []
        fix_f1_l = []

        for fac in self.args.fac_list:
            if not (
                (self.args.hypothesis_type == "P-NNH" and self.args.agg == "pct")
                or (self.args.hypothesis_type == "NNH" and self.args.agg == "avg")
            ):
                print(
                    f"no HT application for {self.args.hypothesis_type} and {self.args.agg}"
                )
                break

            for H1_op in ["greater", "less"]:
                if self.args.hypothesis_type == "P-NNH" and self.args.agg == "pct":
                    c_time_GT = (len(self.args.true_ans_D) / Oracle_dist.shape[0]) * fac

                    _, _, GT = one_proportion_z_test(
                        len(self.args.true_ans_D),
                        Oracle_dist.shape[0],
                        c_time_GT,
                        0.05,
                        H1_op,
                    )

                    rt_align, _ = HT_acc_z_test(
                        self.args,
                        "PQE-RT",
                        ANS,
                        oracle_dist_S.shape[0],
                        GT,
                        c_time_GT,
                        H1_op,
                    )

                    acc_l.append(rt_align)

                elif self.args.hypothesis_type == "NNH" and self.args.agg == "avg":
                    c_time_GT = self.args.agg_D * fac

                    _, GT, _, _ = HT_acc_t_test(
                        self.args, self.args.l_D, c_time_GT, H1_op, is_D=True
                    )
                    rt_align, _, rt_CI_l, rt_CI_h = HT_acc_t_test(
                        self.args, L_S, c_time_GT, H1_op, GT=GT, is_D=False
                    )

                    acc_l.append(rt_align)
                    CI_l.append(rt_CI_h - rt_CI_l)
                    if f1 is not None:
                        f1_l.append(f1)

        return acc_l, CI_l, f1_l, fix_f1_l


def run_experiment(args, Oracle_dist, Proxy_dist, seed):
    """
    Run a complete experiment with multiple samples.

    Args:
        args: Arguments object
        Oracle_dist: Full oracle distances
        Proxy_dist: Full proxy distances
        seed: Random seed

    Returns:
        Experiment results tuple
    """
    runner = ExperimentRunner(args)

    # Initialize result lists
    acc_l = []
    relativeError_l = []
    absoluteError_l = []
    recall_l = []
    precision_l = []
    fix_prec_l = []
    fix_rec_l = []
    agg_l = []
    agg_S_l = []
    NN_S_l = []
    NN_RT_l = []
    time_l = []
    cannot_times_l = []
    prec_rec_diff_l = []

    if args.agg in ["avg", "var", "sum", "min", "max", "median"]:
        CI_l = []
        f1_l = []
        fix_f1_l = []

    for i in range(args.num_sample):
        start_sample = time.time()
        np.random.seed(seed * i)

        # Sample a subset S from the full distributions
        indices = np.random.choice(Oracle_dist.shape[0], args.s, replace=False)
        oracle_dist_S = Oracle_dist[indices]
        proxy_dist_S = Proxy_dist[indices]
        args.true_ans_S = np.where(oracle_dist_S <= args.Dist_t)[0]
        args.NN_S = len(args.true_ans_S)
        verbose_print(
            args, f">>> algo {args.algo} | sample {i} | Find NN in S is {args.NN_S}"
        )

        # Compute Aggregation Value in S
        if args.agg == "pct":
            args.agg_S = len(args.true_ans_S) / oracle_dist_S.shape[0]
            verbose_print(args, f"the prop in S is {args.agg_S}")
        elif args.agg == "count":
            args.agg_S = len(args.true_ans_S)
            verbose_print(args, f"the count in S is {args.agg_S}")
        else:
            args.S_attr = [args.D_attr[i] for i in indices]
            args.l_S, args.agg_S = agg_value(
                args.S_attr, args.true_ans_S, args.attr_id, args.agg
            )
            _, args.agg_S_full = agg_value(
                args.S_attr, range(len(args.S_attr)), args.attr_id, args.agg
            )
            verbose_print(
                args,
                f"The number of NN in S is {len(args.true_ans_S)} ({(len(args.true_ans_S) / proxy_dist_S.shape[0]) * 100}%), the aggregation of true NN is {args.agg_S} and the aggregated value of all data in S is {args.agg_S_full}",
            )

        # Choose pilot samples
        args.fix_sample = np.random.choice(
            len(oracle_dist_S), size=int(args.s_p), replace=False
        )

        args.oracle_dist_S_p = oracle_dist_S[args.fix_sample]
        args.proxy_dist_S_p = proxy_dist_S[args.fix_sample]
        pilot_nn = len(np.where(args.oracle_dist_S_p <= args.Dist_t)[0])
        verbose_print(args, f"Number of NN in pilot sample: {pilot_nn}")

        start_query_sample = time.time()

        # Run the selected algorithm
        prec, rec, optimal_cost, ANS, fix_prec, fix_rec, CANNOT = runner.run_algorithm(
            oracle_dist_S, proxy_dist_S, start_sample
        )

        args.NN = len(ANS)
        prec_rec_diff_l.append(abs(prec - rec))
        verbose_print(
            args,
            f"{args.algo} results | Recall: {rec}, Precision: {prec}, "
            f"Fix Prec: {fix_prec}, Fix Rec: {fix_rec} (Cost: {args.s_p})",
        )

        recall_l.append(rec)
        precision_l.append(prec)
        fix_prec_l.append(fix_prec)
        fix_rec_l.append(fix_rec)

        # Compute the approximated aggregation over found NN
        if args.agg in ["pct", "count"]:
            approx_agg_S, _, _ = runner.compute_aggregation(proxy_dist_S, ANS)
            verbose_print(
                args,
                f"the approx {'prop' if args.agg == 'pct' else 'count'} is {approx_agg_S}",
            )
        else:
            L_S, approx_agg_S = agg_value(args.S_attr, ANS, args.attr_id, args.agg)
            verbose_print(
                args,
                f"The number of NN by {args.algo} is {args.NN} ({(args.NN / proxy_dist_S.shape[0]) * 100}%), the approximated aggregated value is {approx_agg_S}",
            )

            if (prec + rec) == 0:
                f1 = 0
            else:
                f1 = compute_f1_score(args, prec, rec)
            if np.isnan(fix_prec):
                fix_f1 = np.nan
            elif (fix_prec + fix_rec) == 0:
                fix_f1 = 0
            else:
                fix_f1 = compute_f1_score(args, fix_prec, fix_rec)

            f1_l.append(f1)
            fix_f1_l.append(fix_f1)

        time_l.append(round(time.time() - start_query_sample, 2))

        # Hypothesis Testing
        before_HT = time.time()
        acc_results, CI_results, f1_results, fix_f1_results = (
            runner.run_hypothesis_testing(
                Oracle_dist,
                oracle_dist_S,
                ANS,
                L_S if args.agg not in ["pct", "count"] else None,
                f1,
            )
        )
        acc_l.extend(acc_results)
        CI_l.extend(CI_results)
        f1_l.extend(f1_results)
        fix_f1_l.extend(fix_f1_results)

        verbose_print(
            args, f"time of hypothesis testing {round(time.time() - before_HT, 2)}"
        )

        # Error Calculation
        if args.agg == "sum":
            approx_agg_S = approx_agg_S * (Oracle_dist.shape[0] / args.s)
        elif args.agg == "count":
            approx_agg_S = approx_agg_S * (Oracle_dist.shape[0] / args.s)

        if args.agg_D == 0:
            relativeError = float("inf")
        else:
            relativeError = abs(approx_agg_S - args.agg_D) / args.agg_D * 100
        relativeError_l.append(relativeError)
        absoluteError = abs(approx_agg_S - args.agg_D)
        absoluteError_l.append(absoluteError)

        agg_l.append(approx_agg_S)
        agg_S_l.append(args.agg_D)
        NN_S_l.append(args.NN_S)
        NN_RT_l.append(args.NN)
        cannot_times_l.append(CANNOT)

    # Compute Overall Statistics
    avg_acc = np.nanmean(acc_l)
    avg_absError = np.nanmean(absoluteError_l)
    avg_error = np.nanmean(relativeError_l)
    avg_rec = np.nanmean(recall_l)
    avg_prec = np.nanmean(precision_l)
    avg_fix_rec = np.nanmean(fix_rec_l)
    avg_fix_prec = np.nanmean(fix_prec_l)
    avg_NN_S = np.nanmean(NN_S_l)
    avg_NN_RT = np.nanmean(NN_RT_l)
    cannot_times = np.nanmean(cannot_times_l)
    avg_execution_time = np.nanmean(time_l[1:])

    verbose_print(
        args, f"Average relative error over {args.num_sample} runs: {avg_error}"
    )
    verbose_print(
        args, f"Average absolute error over {args.num_sample} runs: {avg_absError}"
    )
    verbose_print(args, f"Average HT accuracy over {args.num_sample} runs: {avg_acc}")
    verbose_print(args, f"Average recall: {avg_rec}")
    verbose_print(args, f"Average precision: {avg_prec}")
    verbose_print(args, f"Average fixed recall: {avg_fix_rec}")
    verbose_print(args, f"Average fixed precision: {avg_fix_prec}")
    verbose_print(args, f"Average NN in S: {avg_NN_S}")
    verbose_print(args, f"Average NN by {args.algo} in S: {avg_NN_RT}")
    verbose_print(args, f"Average execution time: {avg_execution_time}")

    avg_agg = np.nanmean(agg_l)
    var_agg = np.nanvar(agg_l)
    avg_agg_S = np.nanmean(agg_S_l)
    prec_rec_diff = np.nanmean(prec_rec_diff_l)

    # Return Results Based on Aggregation Type
    if args.agg in ["pct", "count"]:
        verbose_print(
            args,
            f"Avg {'prop' if args.agg == 'pct' else 'count'}_S: {avg_agg} with variance {round(var_agg, 4)}",
        )
        return (
            avg_execution_time,
            avg_error,
            avg_absError,
            avg_acc,
            avg_rec,
            avg_prec,
            avg_fix_rec,
            avg_fix_prec,
            avg_NN_RT,
            avg_agg,
            var_agg,
            avg_NN_S,
            avg_agg_S,
            prec_rec_diff,
            None,
            None,
            None,
            cannot_times,
        )
    else:
        avg_CI = np.mean(CI_l)
        avg_f1 = np.mean(f1_l)
        avg_fix_f1 = np.mean(fix_f1_l)
        verbose_print(
            args, f"Avg aggregate value: {avg_agg} with variance {round(var_agg, 4)}"
        )
        verbose_print(args, f"Avg aggregate in S: {avg_agg_S}")
        verbose_print(args, f"Avg CI: {avg_CI}")
        verbose_print(args, f"Avg F1 score: {avg_f1}")
        verbose_print(args, f"Avg fixed F1 score: {avg_fix_f1}")
        return (
            avg_execution_time,
            avg_error,
            avg_absError,
            avg_acc,
            avg_rec,
            avg_prec,
            avg_fix_rec,
            avg_fix_prec,
            avg_NN_RT,
            avg_agg,
            var_agg,
            avg_NN_S,
            avg_agg_S,
            prec_rec_diff,
            avg_CI,
            avg_f1,
            avg_fix_f1,
            None,
        )
