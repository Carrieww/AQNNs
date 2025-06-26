import copy

import numpy as np
import pandas as pd

from util import (
    agg_value,
    create_semi_synthetic_data,
    get_data,
    load_data,
    prepare_distances,
    verbose_print,
)


class DataProcessor:
    """
    Class to handle data loading and preprocessing.
    """

    def __init__(self, args):
        self.args = args

    def load_and_preprocess_data(self):
        """
        Load and preprocess data based on arguments.

        Returns:
            Proxy_emb, Oracle_emb, args with loaded data
        """
        # Load data
        Proxy_emb, Oracle_emb = load_data(self.args)

        # Setup factor list
        self.args.fac_list = np.arange(
            float(self.args.fac_list.split(",")[0]),
            float(self.args.fac_list.split(",")[1]),
            float(self.args.fac_list.split(",")[2]),
        )

        verbose_print(
            self.args,
            f"Prob: {self.args.Prob}; r: {self.args.Dist_t}; beta: {self.args.beta}",
        )
        verbose_print(
            self.args,
            f"Dataset: {self.args.Fname}, size: {Proxy_emb.shape} , aggregation function: {self.args.agg}, filename: {self.args.file_suffix}, s: {self.args.s}, s_p: {self.args.s_p}",
        )

        # Handle hypothesis-specific setup
        self._setup_hypothesis_data()

        # Create semi-synthetic data
        Proxy_emb, Oracle_emb = self._create_synthetic_data(Proxy_emb, Oracle_emb)

        return Proxy_emb, Oracle_emb

    def _setup_hypothesis_data(self):
        """Setup hypothesis-specific data based on aggregation type."""
        if self.args.agg == "pct":
            self.args.attr = "proportion"
        elif self.args.agg == "count":
            self.args.attr = "count"
        elif self.args.agg in ["avg", "var", "sum", "min", "max", "median"]:
            self._load_attribute_data()
            verbose_print(
                self.args,
                f"H1: {self.args.agg} {self.args.attr} (ind = {self.args.attr_id}) of NNs of q is tested.",
            )
        else:
            raise ValueError("Unknown aggregation function specified in args.agg")

    def _load_attribute_data(self):
        """Load attribute data based on dataset."""
        if self.args.Fname in ["eICU", "MIMIC-III"]:
            attr_filename = f"data/Medical/{self.args.Fname}/{self.args.Fname}.testfull"
            self.args.ori_D_attr = get_data(filename=attr_filename)
        elif self.args.Fname in ["Jigsaw"]:
            attr_filename = f"data/{self.args.Fname}/{self.args.Fname}_attribute.csv"
            df = pd.read_csv(attr_filename)
            self.args.ori_D_attr = np.vstack(np.array(df["no_downvote"])).tolist()
            self.args.ori_D_attr = [["", "", item] for item in self.args.ori_D_attr]
        elif self.args.Fname in ["yelp"]:
            attr_filename = f"data/{self.args.Fname}/{self.args.Fname}_test.csv"
            df = pd.read_csv(attr_filename, header=None, names=["rating", "text"])
            self.args.ori_D_attr = np.vstack(np.array(df["rating"])).tolist()
            self.args.ori_D_attr = [["", "", item] for item in self.args.ori_D_attr]
        elif self.args.Fname in ["Electronics"]:
            attr_filename = (
                f"data/Amazon/{self.args.Fname}/{self.args.Fname}_amazon_test.csv"
            )
            df = pd.read_csv(attr_filename, header=None, names=["rating", "text"])
            self.args.ori_D_attr = np.vstack(np.array(df["rating"])).tolist()
            self.args.ori_D_attr = [["", "", item] for item in self.args.ori_D_attr]
        else:
            raise ValueError("Unknown filename")

    def _create_synthetic_data(self, Proxy_emb, Oracle_emb):
        """Create semi-synthetic data based on aggregation type."""
        if self.args.agg in ["pct", "count"]:
            Proxy_emb, Oracle_emb, _ = create_semi_synthetic_data(
                Oracle_emb,
                Proxy_emb,
                [],
                factor=self.args.scalability_factor,
                noise_level=0.001,
            )
        elif self.args.agg in ["avg", "var", "sum", "min", "max", "median"]:
            Proxy_emb, Oracle_emb, self.args.ori_D_attr = create_semi_synthetic_data(
                Oracle_emb,
                Proxy_emb,
                self.args.ori_D_attr,
                factor=self.args.scalability_factor,
                noise_level=0.1,
            )
            verbose_print(
                self.args, f"Scalability factor: {self.args.scalability_factor}"
            )
            verbose_print(
                self.args,
                f"Oracle_emb shape: {Oracle_emb.shape}, Proxy_emb shape: {Proxy_emb.shape}",
            )
        else:
            raise ValueError("Unknown aggregation function specified in args.agg")

        return Proxy_emb, Oracle_emb

    def prepare_query_data(self, Oracle_emb, Proxy_emb, seed, query_index_list=None):
        """
        Prepare query data for experiments.

        Args:
            Oracle_emb: Oracle embeddings
            Proxy_emb: Proxy embeddings
            seed: Random seed
            query_index_list: Optional predefined query indices

        Returns:
            Oracle_dist, Proxy_dist, args with ground truth data
        """
        # Get query indices
        query_indices = (
            np.array([seed])
            if query_index_list is not None
            else np.random.choice(
                range(len(Oracle_emb)), size=self.args.num_query, replace=False
            )
        )

        # Prepare distances
        query_emb = Oracle_emb[query_indices]
        Oracle_dist, Proxy_dist = prepare_distances(
            self.args, Oracle_emb, Proxy_emb, query_emb
        )

        # Calculate ground truth
        self.args.true_ans_D = np.where(Oracle_dist <= self.args.Dist_t)[0]
        self._calculate_ground_truth_aggregation(Oracle_emb, Oracle_dist)

        return Oracle_dist, Proxy_dist

    def _calculate_ground_truth_aggregation(self, Oracle_emb, Oracle_dist):
        """Calculate ground truth aggregation values."""
        if self.args.agg == "pct":
            self.args.agg_D = len(self.args.true_ans_D) / Oracle_dist.shape[0]
            verbose_print(
                self.args,
                f"Ground Truth NN is {len(self.args.true_ans_D)}, proportion is {self.args.agg_D}",
            )
        elif self.args.agg == "count":
            self.args.agg_D = len(self.args.true_ans_D)
            verbose_print(
                self.args,
                f"Ground Truth NN is {len(self.args.true_ans_D)}, count is {self.args.agg_D}",
            )
        elif self.args.agg in ["avg", "var", "sum", "min", "max", "median"]:
            self.args.l_D, self.args.agg_D = agg_value(
                self.args.ori_D_attr,
                self.args.true_ans_D,
                self.args.attr_id,
                self.args.agg,
            )

            l_D_full, agg_D_full = agg_value(
                self.args.ori_D_attr,
                range(len(self.args.ori_D_attr)),
                self.args.attr_id,
                self.args.agg,
            )
            print(f"agg_D: {self.args.agg_D}, agg_D_full: {agg_D_full}")
            self.args.D_attr = copy.deepcopy(self.args.ori_D_attr)

            verbose_print(
                self.args,
                f"Ground Truth NN is {len(self.args.true_ans_D)} ({round(len(self.args.true_ans_D) / len(Oracle_emb), 4) * 100}%), aggregation is {self.args.agg_D}",
            )
        else:
            raise ValueError("Unknown aggregation function specified in args.agg")
