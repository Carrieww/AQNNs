from util import get_data, preprocess_dist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_proxy_quality(oracle, proxy, index, f=''):
    diff_dist = list()
    archive_oracle = list()
    archive_proxy = list()
    for i in range(len(index)):
        proxy_dist, oracle_dist = preprocess_dist(oracle, proxy, oracle[[index[i]]])
        archive_oracle += list(oracle_dist)
        archive_proxy += list(proxy_dist)
        diff_dist += list(np.array(oracle_dist) - np.array(proxy_dist))

    font_size_hp = 18
    label_fs = 20
    plt.rcParams['font.size'] = font_size_hp
    plt.hist(diff_dist, bins=20, density=True)
    # plt.title(r'Histogram of $\epsilon=dist^O(x)-dist^P(x)$', fontsize=label_fs)
    # plt.title('%s' % f, fontsize=label_fs)
    plt.xlabel(r'$dist^O(x_i)-dist^P(x_i)$', fontsize=label_fs)
    plt.ylabel('Frequency (%)', fontsize=label_fs,fontweight='bold')
    plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=font_size_hp)
    plt.yticks(fontsize=font_size_hp)
    plt.tight_layout()
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(6, 4)
    plt.show()

def load_data(name):
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
    elif name == "Jigsaw":
        filename_pred = f"data/{name}/" + name + ".pred"
        filename_truth = f"data/{name}/" + name + ".truth"

        proxy_pred = np.array(get_data(filename=filename_pred))
        oracle_pred = np.array(get_data(filename=filename_truth))

        return proxy_pred, oracle_pred
    elif name == "Jackson":
        filename = f"data/Video/jackson10000_attribute.csv"
        df = pd.read_csv(filename)
        return np.vstack(np.array(df["proxy_score"])), np.vstack(np.array(df["label"]))
    else:
        raise Exception("The dataset is not implemented yet")


if __name__ == '__main__':
    num_query = 100

    for Fname in ["eICU", "MIMIC-III"]:  # "eICU", "MIMIC-III", "Amazon-E", "Amazon-HH", "Jackson", "Jigsaw"
        Proxy, Oracle = load_data(name=Fname)

        np.random.seed(0)
        Index = np.random.choice(range(len(Oracle)), size=num_query, replace=False)

        plot_proxy_quality(oracle=Oracle, proxy=Proxy, index=Index)






