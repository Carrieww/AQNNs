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
        elif self.args.algo in ["PQE-RT", "PQE-PT"]:
            return self._run_pqe(oracle_dist_S, proxy_dist_S, start_sample_time)
        elif self.args.algo in ["SUPG-RT", "SUPG-PT"]:
            return self._run_supg(oracle_dist_S, proxy_dist_S, start_sample_time)
        elif self.args.algo == "TopK":
            return self._run_topk(oracle_dist_S, proxy_dist_S, start_sample_time)
        else:
            raise ValueError(f"Unknown algorithm: {self.args.algo}")

    def _run_sprint(self, oracle_dist_S, proxy_dist_S, start_sample_time):
        """Run SPRinT algorithm."""
        before_pt = time.time()
        verbose_print(
            self.args,
            f"Data preparation time: {round(time.time() - start_sample_time, 2)} sec",
        )

        prec, rec, _, ANS, fix_prec, fix_rec, CANNOT = (
            self.sprint_runner.run_single_experiment(oracle_dist_S, proxy_dist_S)
        )

        verbose_print(
            self.args,
            f"Optimal pt search time: {round(time.time() - before_pt, 2)} sec",
        )

        sample_time = round(time.time() - start_sample_time, 2)

        return prec, rec, self.args.s_p, ANS, fix_prec, fix_rec, CANNOT, sample_time

    def _run_pqe(self, oracle_dist_S, proxy_dist_S, start_sample_time):
        """Run PQE algorithm."""
        self.args.optimal_cost = self.args.s_p
        self.args.target = (
            self.args.recall_target
            if self.args.algo == "PQE-RT"
            else self.args.precision_target
        )
        prec, rec, _, ANS, _, _ = test_PQE(
            self.args,
            oracle_dist_S,
            proxy_dist_S,
            self.args.algo[-2:],
            self.args.target,
        )
        sample_time = round(time.time() - start_sample_time, 2)
        fix_prec, fix_rec, CANNOT = np.nan, np.nan, np.nan
        return prec, rec, self.args.s_p, ANS, fix_prec, fix_rec, CANNOT, sample_time

    def _run_supg(self, oracle_dist_S, proxy_dist_S, start_sample_time):
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
        sample_time = round(time.time() - start_sample_time, 2)
        fix_prec, fix_rec, CANNOT = np.nan, np.nan, np.nan
        return prec, rec, self.args.s_p, ANS, fix_prec, fix_rec, CANNOT, sample_time

    def _run_topk(self, oracle_dist_S, proxy_dist_S, start_sample_time):
        """Run TopK algorithm."""
        prec, rec, optimal_cost, ANS = test_topk(
            oracle_dist=oracle_dist_S,
            proxy_dist=proxy_dist_S,
            scale=norm_scale,
            t=self.args.Dist_t,
            prob=self.args.Prob,
        )
        sample_time = round(time.time() - start_sample_time, 2)
        fix_prec, fix_rec, CANNOT = np.nan, np.nan, np.nan
        return prec, rec, optimal_cost, ANS, fix_prec, fix_rec, CANNOT, sample_time

    def compute_aggregation_for_S(self, agg, oracle_dist_S, proxy_dist_S, indices):
        """Compute aggregation value in S for a specific aggregation type"""
        if agg == "pct":
            agg_S = len(self.args.true_ans_S) / oracle_dist_S.shape[0]
            verbose_print(self.args, f"the prop in S is {agg_S}")
            return agg_S, None
        else:
            self.args.S_attr = [self.args.D_attr[i] for i in indices]
            self.args.l_S, agg_S = agg_value(
                self.args.S_attr, self.args.true_ans_S, self.args.attr_id, agg
            )
            _, agg_S_full = agg_value(
                self.args.S_attr, range(len(self.args.S_attr)), self.args.attr_id, agg
            )
            verbose_print(
                self.args,
                f"The number of NN in S is {len(self.args.true_ans_S)} ({(len(self.args.true_ans_S) / proxy_dist_S.shape[0]) * 100}%), the aggregation of true NN is {agg_S} and the aggregated value of all data in S is {agg_S_full}",
            )
            return agg_S, agg_S_full

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
    results = {}
    for agg in args.agg_types:
        results[agg] = {
            'acc_l': [],
            'relativeError_l': [],
            'absoluteError_l': [],
            'recall_l': [],
            'precision_l': [],
            'fix_prec_l': [],
            'fix_rec_l': [],
            'agg_l': [],
            'agg_S_l': [],
            'NN_S_l': [],
            'NN_algo_l': [],
            'time_l': [],
            'cannot_times_l': [],
            'prec_rec_diff_l': [],
            'f1_l': [],
            'fix_f1_l': [],
            'CI_l': []
        }
    
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
            args, f">>> algo {args.algo} | sample {i} | Find NN in S is {args.NN_S} ({args.NN_S / oracle_dist_S.shape[0] * 100}%)"
        )

        # Choose pilot samples
        args.fix_sample = np.random.choice(
            len(oracle_dist_S), size=int(args.s_p), replace=False
        )

        args.oracle_dist_S_p = oracle_dist_S[args.fix_sample]
        args.proxy_dist_S_p = proxy_dist_S[args.fix_sample]
        pilot_nn = len(np.where(args.oracle_dist_S_p <= args.Dist_t)[0])
        verbose_print(args, f"Number of NN in pilot sample: {pilot_nn}")

        # Run the selected algorithm
        prec, rec, _, ANS, fix_prec, fix_rec, CANNOT, sample_time = runner.run_algorithm(
            oracle_dist_S, proxy_dist_S, start_sample
        )

        for agg in args.agg_types:
            args.agg_S, args.agg_S_full = runner.compute_aggregation_for_S(
                agg, oracle_dist_S, proxy_dist_S, indices
            )
            process_aggregation_results(args,results[agg], ANS, prec, rec, fix_prec, fix_rec,
                oracle_dist_S, proxy_dist_S, CANNOT, sample_time, Oracle_dist.shape[0], agg
            )
    # --- Compute Overall Statistics for each aggregation type ---
    final_results = {}
    for agg in args.agg_types:
        result_dict = results[agg]
        
        avg_acc = np.nanmean(result_dict['acc_l']) if result_dict['acc_l'] else np.nan
        avg_absError = np.nanmean(result_dict['absoluteError_l'])
        avg_error = np.nanmean(result_dict['relativeError_l'])
        avg_rec = np.nanmean(result_dict['recall_l'])
        avg_prec = np.nanmean(result_dict['precision_l'])
        avg_fix_rec = np.nanmean(result_dict['fix_rec_l'])
        avg_fix_prec = np.nanmean(result_dict['fix_prec_l'])
        avg_NN_S = np.nanmean(result_dict['NN_S_l'])
        avg_NN_algo = np.nanmean(result_dict['NN_algo_l'])
        cannot_times = np.nanmean(result_dict['cannot_times_l'])
        avg_execution_time = np.nanmean(result_dict['time_l'][1:]) if len(result_dict['time_l']) > 1 else np.nanmean(result_dict['time_l'])

        verbose_print(
            args, f"[{agg}] Average relative error over {args.num_sample} runs: {avg_error}"
        )
        verbose_print(
            args, f"[{agg}] Average absolute error over {args.num_sample} runs: {avg_absError}"
        )
        verbose_print(args, f"[{agg}] Average HT accuracy over {args.num_sample} runs: {avg_acc}")
        verbose_print(args, f"[{agg}] Average recall: {avg_rec}")
        verbose_print(args, f"[{agg}] Average precision: {avg_prec}")
        verbose_print(args, f"[{agg}] Average fixed recall: {avg_fix_rec}")
        verbose_print(args, f"[{agg}] Average fixed precision: {avg_fix_prec}")
        verbose_print(args, f"[{agg}] Average NN in S: {avg_NN_S}")
        verbose_print(args, f"[{agg}] Average NN by {args.algo} in S: {avg_NN_algo}")
        verbose_print(args, f"[{agg}] Average execution time: {avg_execution_time}")

        avg_agg = np.nanmean(result_dict['agg_l'])
        var_agg = np.nanvar(result_dict['agg_l'])
        avg_agg_S = np.nanmean(result_dict['agg_S_l'])
        prec_rec_diff = np.nanmean(result_dict['prec_rec_diff_l'])

        # --- Return Results Based on Aggregation Type ---
        if agg in ["pct"]:
            verbose_print(args, f"[{agg}] Avg prop_S: {avg_agg} with variance {round(var_agg, 4)}")
            final_results[agg] = (
                avg_execution_time,
                avg_error,
                avg_absError,
                avg_acc,
                avg_rec,
                avg_prec,
                avg_fix_rec,
                avg_fix_prec,
                avg_NN_algo,
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
            avg_CI = np.mean(result_dict['CI_l']) if result_dict['CI_l'] else np.nan
            avg_f1 = np.mean(result_dict['f1_l']) if result_dict['f1_l'] else np.nan
            avg_fix_f1 = np.mean(result_dict['fix_f1_l']) if result_dict['fix_f1_l'] else np.nan
            verbose_print(
                args, f"[{agg}] Avg aggregate value: {avg_agg} with variance {round(var_agg, 4)}"
            )
            verbose_print(args, f"[{agg}] Avg aggregate in S: {avg_agg_S}")
            verbose_print(args, f"[{agg}] Avg CI: {avg_CI}")
            verbose_print(args, f"[{agg}] Avg F1 score: {avg_f1}")
            verbose_print(args, f"[{agg}] Avg fixed F1 score: {avg_fix_f1}")
            final_results[agg] = (
                avg_execution_time,
                avg_error,
                avg_absError,
                avg_acc,
                avg_rec,
                avg_prec,
                avg_fix_rec,
                avg_fix_prec,
                avg_NN_algo,
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
    
    return final_results

def process_aggregation_results(args, result_dict, ANS, prec, rec, fix_prec, fix_rec, oracle_dist_S, proxy_dist_S, CANNOT, sample_time, oracle_dist_shape, agg):
    """Process results for a specific aggregation type"""
    verbose_print(args, f"----- Processing results for {agg} -----")
    args.NN = len(ANS)
    result_dict['prec_rec_diff_l'].append(abs(prec-rec))
    
    verbose_print(
        args,
        f"{args.algo} results | Recall: {rec}, Precision: {prec}, "
        f"Fix Prec: {fix_prec}, Fix Rec: {fix_rec} (Cost: {args.s_p})",
    )

    result_dict['recall_l'].append(rec)
    result_dict['precision_l'].append(prec)
    result_dict['fix_prec_l'].append(fix_prec)
    result_dict['fix_rec_l'].append(fix_rec)

    # --- Compute the approximated aggregation over found NN ---
    if agg == "pct":
        approx_agg_S = round(args.NN / proxy_dist_S.shape[0], 4)
        verbose_print(args, f"the approx prop is {approx_agg_S}")
        f1 = 0
        fix_f1 = 0
        L_S = None  # Not used for pct
    else:
        L_S, approx_agg_S = agg_value(args.S_attr, ANS, args.attr_id, agg)
        verbose_print(args, f"the approximated {agg} is {approx_agg_S}")

        # --- Error Calculation ---
        if agg == "sum":
            approx_agg_S = approx_agg_S * (oracle_dist_shape / args.s)

        verbose_print(
            args,
            f"The number of NN by {args.algo} is {args.NN} ({(args.NN / proxy_dist_S.shape[0]) * 100}%), the approximated {agg} is {approx_agg_S}",
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

    result_dict['f1_l'].append(f1)
    result_dict['fix_f1_l'].append(fix_f1)

    result_dict['time_l'].append(sample_time)

    # --- Hypothesis Testing ---
    before_HT = time.time()
    for fac in args.fac_list:
        if not (
                (agg == "pct")
                or (agg == "avg")
        ):
            verbose_print(args, f"no HT application")
            break
        for H1_op in ["greater", "less"]:
            if agg == "pct":
                c_time_GT = (len(args.true_ans_D) / oracle_dist_shape) * fac

                _, _, GT = one_proportion_z_test(
                    len(args.true_ans_D),
                    oracle_dist_shape,
                    c_time_GT,
                    0.05,
                    H1_op,
                )

                rt_align, _ = HT_acc_z_test(
                    ANS,
                    oracle_dist_S.shape[0],
                    GT,
                    c_time_GT,
                    H1_op,
                )

                result_dict['acc_l'].append(rt_align)

            elif agg == "avg":
                c_time_GT = args.agg_D_dict[agg] * fac

                _, GT, _, _ = HT_acc_t_test(
                    args.l_D_dict[agg], c_time_GT, H1_op, is_D=True
                )
                if L_S is not None:
                    rt_align, _, rt_CI_l, rt_CI_h = HT_acc_t_test(
                        L_S, c_time_GT, H1_op, GT=GT, is_D=False
                    )
                    result_dict['acc_l'].append(rt_align)
                    result_dict['CI_l'].append(rt_CI_h - rt_CI_l)

    verbose_print(
        args, f"time of hypothesis testing {round(time.time() - before_HT, 2)}"
    )

    # Use the correct ground truth value for this aggregation type
    agg_D_value = args.agg_D_dict[agg]
    if agg_D_value == 0:
        relativeError = float('inf')
    else:
        relativeError = abs(approx_agg_S - agg_D_value) / agg_D_value * 100
    result_dict['relativeError_l'].append(relativeError)
    absoluteError = abs(approx_agg_S - agg_D_value)
    result_dict['absoluteError_l'].append(absoluteError)

    result_dict['agg_l'].append(approx_agg_S)
    result_dict['agg_S_l'].append(agg_D_value)
    result_dict['NN_S_l'].append(args.NN_S)
    result_dict['NN_algo_l'].append(args.NN)
    result_dict['cannot_times_l'].append(CANNOT)
