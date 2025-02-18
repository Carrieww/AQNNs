import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).parent


def parse_args():
    parser = argparse.ArgumentParser(description="Model parameters")
    parser.add_argument("--s_p", type=int, default=600, help="Pilot sample size.")
    parser.add_argument("--s", type=int, default=1000, help="Sample size.")
    parser.add_argument(
        "--hypothesis_type",
        type=str,
        default="NNH",
        choices=["NNH", "P-NNH"],
        help="For AVG, NNH is recommended. For PCT, P-NNH is recommended.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="SPRinT",
        choices=[
            "SPRinT",
            "PQA-PT",
            "PQA-RT",
            "SUPG-PT",
            "SUPG-RT",
            "TopK",
        ],
        help="Choose an algorithm.",
    )
    parser.add_argument(
        "--agg",
        type=str,
        default="avg",
        choices=["avg", "pct", "var", "sum"],
        help="Choose an aggregation function.",
    )
    parser.add_argument(
        "--attr", type=str, default="age", help="attribute name in the hypothesis."
    )
    parser.add_argument(
        "--attr_id",
        type=int,
        default=1,
        help="the id of the attribute stored in the database.",
    )
    parser.add_argument(
        "--file_suffix", type=str, default="test", help="log filename."
    )
    parser.add_argument(
        "--Fname",
        type=str,
        default="eICU",
        choices=[
            "eICU",
            "MIMIC-III",
            "Amazon-HH",
            "Amazon-E",
            "Jackson",
        ],
        help="Choose a dataset.",
    )
    parser.add_argument("--num_query", type=int, default=1, help="Number of queries.")
    parser.add_argument("--beta", type=float, default=1, help="Fbeta score coefficient")
    parser.add_argument("--num_sample", type=int, default=30, help="Number of samples.")
    parser.add_argument("--Dist_t", type=float, default=0.9, help="Distance threshold.")
    parser.add_argument(
        "--Prob", type=float, default=0.9, help="Probability threshold."
    )
    parser.add_argument(
        "--recall_target",
        type=float,
        default=0.95,
        help="Recall target for recall target queries.",
    )
    parser.add_argument(
        "--precision_target",
        type=float,
        default=0.95,
        help="Precision target fpr precision target queries.",
    )

    # Hypothesis testing parameters
    parser.add_argument(
        "--fac_list",
        type=str,
        default="0.5,1.51,0.05",
        help="Factor of the constant in a one-sample hypothesis testing.",
    )

    parser.add_argument("--verbose", type=bool, default=True, help="Allow print.")

    args = parser.parse_args()
    return args
