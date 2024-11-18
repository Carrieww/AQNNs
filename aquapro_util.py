import numpy as np
from hyper_parameter import norm_scale
import pickle
import scipy
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import truncnorm, norm

from numba import njit
import seaborn as sns

sns.set_theme(style="ticks")


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
    if name in ["icd9_eICU", "icd9_mimic"]:
        filename_pred = f"data/medical/{name}/" + name + ".pred"
        filename_truth = f"data/medical/{name}/" + name + ".truth"

        proxy_pred = np.array(get_data(filename=filename_pred))
        oracle_pred = np.array(get_data(filename=filename_truth))

        return proxy_pred, oracle_pred


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


@njit
def preprocess_ranks(proxy_dist):
    rank2pd = sorted(enumerate(proxy_dist), key=lambda x: x[1])
    ranks = np.array([i[0] for i in rank2pd])

    return ranks


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


def plot_statics(oracle_dist, proxy_dist, sync_oracle, t, norm_scale, f):
    sort_pd = sorted(enumerate(proxy_dist), key=lambda x: x[1])
    pd_rank = [i[0] for i in sort_pd]
    idx = [i for i in range(len(pd_rank))]
    precis_list = list()
    recall_list = list()
    sync_recalls = list()
    sync_preciss = list()
    true_pos = 0
    all_pos = len(np.where(oracle_dist <= t)[0])
    sync_pos = len(np.where(sync_oracle <= t)[0])
    sync_tp = 0
    for j in range(len(pd_rank)):
        if oracle_dist[pd_rank[j]] <= t:
            true_pos += 1
        if sync_oracle[pd_rank[j]] <= t:
            sync_tp += 1
        precis_list.append(true_pos / (j + 1))
        recall_list.append(true_pos / all_pos)
        sync_recalls.append(sync_tp / sync_pos)
        sync_preciss.append(sync_tp / (j + 1))
    plt.xlabel("objects")
    plt.title("precis/recall and proxy_dist, sigma=%.2f (%s)" % (norm_scale, f))
    plt.axhline(y=t, color="k")
    # plt.plot(idx, pd_val, label='proxy_dist')
    plt.plot(idx, precis_list, label="precis")
    plt.plot(idx, recall_list, label="recall")
    # plt.plot(idx, sync_recalls, label='sync_recall')
    # plt.plot(idx, sync_preciss, label='sync_precis')
    plt.legend()
    plt.show()


@njit
def array_union(l1, l2):
    return np.unique(np.concatenate((l1, l2)))


@njit
def set_diff(l1, l2):
    l3 = np.array([i for i in l1 if i not in l2])
    return np.unique(l3)


def prepare_distances(args, Oracle_emb, Proxy_emb, query_indices):
    """Prepare distances using the specified preprocessing method."""
    if args.PQA == "PQA":
        Proxy_dist, _ = preprocess_dist(
            Oracle_emb, Proxy_emb, Oracle_emb[query_indices]
        )
        Oracle_dist = preprocess_sync(Proxy_dist, norm_scale)
    elif args.PQA == "PQE":
        Proxy_dist, Oracle_dist = preprocess_dist(
            Oracle_emb, Proxy_emb, Oracle_emb[query_indices]
        )
    else:
        raise ValueError(f"Invalid PQA: {args.PQA}")
    return Oracle_dist, Proxy_dist


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
        if len(l) == 0:
            res = np.nan
        else:
            res = sum(l) / len(l)
    else:
        raise Exception(f"The case for {agg} has not been implemented yet")
    return l, res


def verbose_print(args, *messages):
    """Print messages if verbose mode is enabled."""
    if args.verbose:
        print(*messages)
