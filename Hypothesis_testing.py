from scipy.stats import norm
from scipy import stats
import numpy as np
from aquapro_util import verbose_print


def HT_acc_z_test(args, name, ans, total, GT, prop_c, H1_op):
    # verbose_print(args, f"start {name} algorithm for q")
    # verbose_print(args, f"FRNN result: {len(ans)}")
    # verbose_print(args, f"total patients: {total}")
    # verbose_print(args, f"c proportion: {prop_c}; approx: {len(ans) / total}")

    z_stat, p_value, reject = one_proportion_z_test(
        len(ans), total, prop_c, 0.05, H1_op
    )

    # verbose_print(
    #     args,
    #     f"Z-Statistic: {z_stat}, P-Value: {p_value}, Reject Null Hypothesis: {reject}",
    # )

    align = reject == GT
    # verbose_print(args, f"align: {align}")

    return align, reject


def one_proportion_z_test(
    successes, total_trials, null_prop, alpha=0.05, alternative="two-sided"
):
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


def HT_acc_t_test(args, l, c, operator, GT=None, is_D=False):
    t_stat, p_value, rejectH0, CI_l, CI_h = one_sample_t_test(
        args, l, c, alternative=operator
    )
    # verbose_print(args, f"t_stat: {t_stat}, p_value: {p_value}")

    if is_D:
        align = True
        # verbose_print(args, f"The ans in D to reject H0 result is : {rejectH0}")

    else:
        assert GT is not None, "GT is None"
        align = rejectH0 == GT
        # verbose_print(args, f"The ans in S to reject H0 result is : {rejectH0}")

    return align, rejectH0, CI_l, CI_h


def one_sample_t_test(args, l, c, alpha=0.05, alternative="two-sided"):
    t_stat, p_value = stats.ttest_1samp(l, popmean=c, alternative=alternative)
    CI_lower, CI_upper = stats.t.interval(
        confidence=1 - alpha,
        df=len(l) - 1,
        loc=np.nanmean(l),
        scale=stats.sem(l),
    )
    rejectH0 = p_value < alpha
    verbose_print(
        args,
        f"The test (c = {c}, op = {alternative}) is significant, we shall {rejectH0} reject the null hypothesis.",
    )
    verbose_print(
        args, f"confidence interval is ({round(CI_lower,4), round(CI_upper,4)})"
    )
    return t_stat, p_value, rejectH0, CI_lower, CI_upper
