import numpy as np
from scipy import stats
from scipy.stats import norm

from util import verbose_print


def HT_acc_z_test(ans, total, GT, prop_c, H1_op):
    z_stat, p_value, reject = one_proportion_z_test(
        len(ans), total, prop_c, 0.05, H1_op
    )
    align = reject == GT
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


def HT_acc_t_test(l, c, operator, GT=None, is_D=False):
    t_stat, p_value, rejectH0, CI_l, CI_h = one_sample_t_test(
        l, c, alternative=operator
    )

    if is_D:
        align = True

    else:
        assert GT is not None, "GT is None"
        align = rejectH0 == GT

    return align, rejectH0, CI_l, CI_h


def one_sample_t_test(l, c, alpha=0.05, alternative="two-sided"):
    t_stat, p_value = stats.ttest_1samp(l, popmean=c, alternative=alternative)
    CI_lower, CI_upper = stats.t.interval(
        confidence=1 - alpha,
        df=len(l) - 1,
        loc=np.nanmean(l),
        scale=stats.sem(l),
    )
    rejectH0 = p_value < alpha
    return t_stat, p_value, rejectH0, CI_lower, CI_upper
