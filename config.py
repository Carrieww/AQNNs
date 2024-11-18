import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).parent


def parse_args():
    parser = argparse.ArgumentParser(description="Model parameters")

    parser.add_argument(
        "--PQA", type=str, default="PQE", help="Whether PQA or PQE is satisfied."
    )
    parser.add_argument(
        "--initial_cost", type=int, default=600, help="Pilot Sample Size."
    )  # s_p
    parser.add_argument("--total_cost", type=int, default=1000, help="Total cost.")  # s
    parser.add_argument(
        "--hypothesis_type",
        type=str,
        default="NNH",
        choices=["NNH", "P-NNH"],
        help="Choose dataset from NNH or P-NNH.",
    )
    parser.add_argument(
        "--attr", type=str, default="age", help="attribute in the hypothesis."
    )
    parser.add_argument(
        "--attr_id",
        type=int,
        default=1,
        help="the id of the attribute in the hypothesis.",
    )
    parser.add_argument(
        "--version", type=str, default="version-600", help="log filename."
    )
    # parser.add_argument(
    #     "--H1_op", type=str, default="less", help="Hypothesis operation."
    # )
    parser.add_argument(
        "--Fname",
        type=str,
        default="icd9_eICU",
        choices=["icd9_eICU", "icd9_mimic"],
        help="Choose dataset.",
    )

    # Model parameters
    parser.add_argument("--num_query", type=int, default=1, help="Number of queries.")
    parser.add_argument("--num_sample", type=int, default=30, help="Number of samples.")
    parser.add_argument("--Dist_t", type=float, default=0.5, help="Distance threshold.")
    parser.add_argument(
        "--Prob", type=float, default=0.95, help="Probability threshold."
    )
    parser.add_argument(
        "--recall_target", type=float, default=0.9, help="Target recall."
    )
    parser.add_argument(
        "--precision_target", type=float, default=0.9, help="Target precision."
    )

    # Cost and sampling

    parser.add_argument(
        "--cost_step_size", type=int, default=10, help="Step size for cost."
    )

    # Fac list and hypothesis testing options
    parser.add_argument(
        "--fac_list", type=str, default="0.5,1.51,0.05", help="Factor list in range."
    )

    parser.add_argument("--verbose", type=bool, default=False, help="Allow print.")

    # RT PT
    parser.add_argument(
        "--rt", type=float, default=0.1, help="recall target t for PQE."
    )

    args = parser.parse_args()
    return args
