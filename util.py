import scipy
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit
from hyper_parameter import norm_scale
from scipy.stats import truncnorm, norm
from scipy.spatial.distance import cdist

sns.set_theme(style="ticks")


def compute_f1_score(args, p, r):
    if p == 0 and r == 0:
        return 0
    else:
        return (1 + args.beta**2) * p * r / ((args.beta**2) * p + r)


def nan2mean(dist):
    mean_val = np.nanmean(dist)
    indices = np.where(np.isnan(dist))
    dist[indices] = mean_val


def get_data(filename=None, is_text=False):
    if is_text:
        instance_list = (np.loadtxt(filename)).astype(float)
    else:
        instance_list = pickle.load(open(filename, "rb"), encoding="latin1")

    return instance_list


def load_data(args):
    name = args.Fname
    if name in ["eICU", "MIMIC-III"]:
        filename_pred = f"data/Medical/{name}/" + name + ".pred"
        filename_truth = f"data/Medical/{name}/" + name + ".truth"

        proxy_pred = np.array(get_data(filename=filename_pred))
        oracle_pred = np.array(get_data(filename=filename_truth))

        return proxy_pred, oracle_pred
    elif name in ["Amazon-HH", "Amazon-E"]:
        filename_pred = f"data/Amazon/{name}/" + name + ".pred"
        filename_truth = f"data/Amazon/{name}/" + name + ".truth"

        proxy_pred = np.array(get_data(filename=filename_pred))
        # proxy_pred = np.array([item[1] for item in proxy_pred])
        oracle_pred_data = get_data(filename=filename_truth)
        oracle_pred = [item[1] for item in oracle_pred_data]
        oracle_pred = np.array(oracle_pred)
        return proxy_pred, oracle_pred
    else:
        filename = f"data/{name}/jackson10000_attribute.csv"

        df = pd.read_csv(filename)

        return np.vstack(np.array(df["proxy_score"])), np.vstack(np.array(df["label"]))


def preprocess_dist(oracle, proxy, query):
    if len(oracle[0]) == 1:
        query = np.array([[1]])
        oracle_dist = cdist(query, oracle, metric="cityblock")[0]
        proxy_dist = cdist(query, proxy, metric="cityblock")[0]
    else:
        oracle_dist = cdist(query, oracle, metric="cosine")[0]
        proxy_dist = cdist(query, proxy, metric="cosine")[0]
    nan2mean(proxy_dist)
    nan2mean(oracle_dist)

    return proxy_dist, oracle_dist


#
# @njit
# def preprocess_ranks(proxy_dist):
#     rank2pd = sorted(enumerate(proxy_dist), key=lambda x: x[1])
#     ranks = np.array([i[0] for i in rank2pd])
#
#     return ranks


def preprocess_topk_phi(proxy_dist, norm_scale, t):
    # This code finds the percentile rank of value t in a normal distribution with mean proxy_dist and standard deviation norm_scale.
    norm_cdfs = norm.cdf(x=t, loc=proxy_dist, scale=norm_scale)
    topk2phi = sorted(enumerate(norm_cdfs), key=lambda x: x[1], reverse=True)
    topk = np.array([i[0] for i in topk2phi])
    phi = np.array([i[1] for i in topk2phi])

    return topk, phi


def preprocess_sync(proxy_dist, norm_scale):
    # it generates random samples from a normal distribution centered around proxy_dist with standard deviation norm_scale.
    # then it clip the samples to be between 0 and 1.
    sync_oracle = np.clip(
        [scipy.stats.norm.rvs(loc=_, scale=norm_scale) for _ in proxy_dist],
        a_min=0,
        a_max=1,
    )

    return sync_oracle


@njit
def array_union(l1, l2):
    return np.unique(np.concatenate((l1, l2)))


@njit
def set_diff(l1, l2):
    l3 = np.array([i for i in l1 if i not in l2])
    return np.unique(l3)


def prepare_distances(args, Oracle_emb, Proxy_emb, query_emb):
    """Prepare distances using the specified preprocessing method."""
    Proxy_dist, _ = preprocess_dist(Oracle_emb, Proxy_emb, query_emb)
    Oracle_dist = preprocess_sync(Proxy_dist, norm_scale)
    return Oracle_dist, Proxy_dist


def is_int(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def agg_value(D, ind_list, attr_id, agg):
    l = []
    for ans_id in ind_list:
        value = D[ans_id][2][attr_id]
        if is_int(value):
            value = float(value)
            if not np.isnan(value):
                l.append(int(value))
        else:
            pass

    if agg == "avg":
        res = np.nan if len(l) == 0 else sum(l) / len(l)
    elif agg == "sum":
        res = np.nan if len(l) == 0 else sum(l)
    elif agg == "var":
        if len(l) == 0:
            res = np.nan
        else:
            mean = sum(l) / len(l)
            res = sum((x - mean) ** 2 for x in l) / len(l)  # Population variance
    else:
        raise Exception(f"The case for {agg} has not been implemented yet")
    return l, res


def verbose_print(args, *messages):
    """Print messages if verbose mode is enabled."""
    if args.verbose:
        print(*messages)


def output_results(
    args,
    seed,
    avg_execution_time,
    avg_error,
    avg_absError,
    avg_NN_RT,
    avg_agg,
    var_agg,
    avg_NN_S,
    avg_agg_S,
    standard_error,
    avg_CI,
    avg_f1,
    avg_fix_f1,
    avg_acc,
    avg_rec,
    avg_prec,
    avg_fix_rec,
    avg_fix_prec,
    cannot_times,
):
    # Saving result
    file_name = f"results/{args.algo}/{args.agg}_{args.Fname}_{args.file_suffix}.txt"
    print(file_name)
    with open(file_name, "a") as file:
        # Write the header (if it's not already present in the file)
        if seed == 1:
            file.write(
                "seed\toptimal cost\tno optimal rt counts\tavg relative error\tavg absolute error\tavg acc\tavg recall\tavg precision\tavg f1\tavg fix recall\tavg fix precision\tavg fix f1\tagg ours\tvar ours\tNN ours\tagg_D\tstandard error\tNN S\tavg CI\tavg exec time\n"
            )

        file.write(
            f"{seed:}\t{args.optimal_cost}\t{cannot_times}\t{avg_error}\t{avg_absError}\t{avg_acc}\t{avg_rec}\t{avg_prec}\t{avg_f1}\t{avg_fix_rec}\t{avg_fix_prec}\t{avg_fix_f1}\t{avg_agg}\t{var_agg}\t{avg_NN_RT}\t{avg_agg_S}\t{standard_error}\t{avg_NN_S}\t{avg_CI}\t{avg_execution_time}\n"
        )

    verbose_print(
        args,
        f"avg error: {avg_error}, avg abs error: {avg_absError}, avg HT acc: {avg_acc}, avg recall: {avg_rec}, avg precision: {avg_prec}, avg fix recall: {avg_fix_rec}, avg precision: {avg_fix_prec}",
    )
    verbose_print(args, "execution time is %.2fs" % (time.time() - args.start_time))
