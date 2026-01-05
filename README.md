# On Efficient Approximate Aggregate Nearest Neighbor Queries over Learned Representations

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the implementation and experimental data for the paper **On Efficient Approximate Aggregate Nearest Neighbor
Queries over Learned Representations**, accepted to **SIGMOD 2026**.


## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Algorithms](#algorithms)
- [Datasets](#datasets)
- [Results](#results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Citation](#citation)

## 🎯 Problem Statement

Given a query point `q` and a radius `r`, we want to compute aggregation functions (e.g., average, variance, sum, proportion) over the attributes of points that are true nearest neighbors of `q` within `r`. 

The challenge is to efficiently compute these aggregations using a combination of **oracle embeddings** (accurate but expensive) and **proxy embeddings** (fast but less accurate) to balance accuracy and computational efficiency.

## 🔬 Algorithms

This repository implements several algorithms for solving Aggregation Queries over Predicted Nearest Neighbors (AQNNs):

### Our framework

![Framework](img/framework.png)

### Our algorithms

- **SPRinT-C** (SPRinT Count-sensitive): Our approach optimized for count-sensitive AQNNs (e.g., proportion queries)
- **SPRinT-V** (SPRinT Value-sensitive): Our approach optimized for value-sensitive AQNNs (e.g., average, variance, sum queries)

## 📊 Datasets

The experiments use the following real-world datasets:

### Medical Datasets
- **eICU**: Electronic Intensive Care Unit dataset
- **MIMIC-III**: Medical Information Mart for Intensive Care III dataset

### E-commerce Datasets
- **Yelp**: Yelp review dataset
- **Electronics**: Amazon Electronics product reviews

### Social Media Datasets
- **Jigsaw**: Jigsaw toxic comment classification dataset

Each dataset includes embeddings (proxy and oracle) and associated attributes for aggregation queries.

## 📈 Results

![Embedding Generation Cost](img/embedding_generation_cost.png)

SPRinT achieves 4.5–186.4× speedup by using proxy models for a small fraction of objects to avoid the majority of expensive oracle calls.

![Relative Error Performance](img/RE_performance.png)

SPRinT-C consistently achieves the lowest relative error (RE) across all datasets. The Two-Phase strategy (combining SPRinT-V and SPRinT-C) also consistently achieves the lowest RE across all datasets, while SPRinT-V performs best on AVG and VAR aggregations for Amazon-E.

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Carrieww/AQNNs.git
   cd AQNNs
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Dependencies

The project requires the following Python packages (see `requirements.txt`):

- `numpy==1.26.4`
- `numba==0.59.1`
- `pandas==2.2.2`
- `scipy==1.13.0`
- `seaborn==0.13.2`
- `feather-format==0.4.1`
- `tqdm==4.66.2`

## 🏃 Quick Start

### Running a Single Experiment

```bash
# Activate virtual environment (if using venv)
source venv/bin/activate

# Run SPRinT-V algorithm for all aggregation functions on MIMIC-III
bash ./myscript.sh
```


## ⚙️ Configuration

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--algo` | str | `SPRinT-V` | Algorithm to use: `SPRinT-C`, `SPRinT-V`, `PQA-PT`, `PQA-RT`, `SUPG-PT`, `SUPG-RT`, `TopK` |
| `--agg` | str | `all` | Aggregation function: `avg`, `var`, `sum`, `pct`, or `all` |
| `--Fname` | str | `MIMIC-III` | Dataset name: `eICU`, `MIMIC-III`, `Jigsaw`, `yelp`, `Electronics` |
| `--s` | int | `500` | Sample size |
| `--s_p` | int | `150` | Pilot sample size |
| `--Dist_t` | float | `0.6` | Distance threshold |
| `--Prob` | float | `0.9` | Probability threshold |
| `--num_query` | int | `1` | Number of queries |
| `--num_sample` | int | `2` | Number of samples |
| `--attr_id` | int | `0` | Attribute ID in the database |
| `--verbose` | bool | `True` | Enable verbose output |

For a complete list of parameters, run:
```bash
python main.py --help
```

### Output

Results are saved in the `results/` directory, organized by algorithm, such as `results/SPRinT-C/`.

Each result file contains detailed metrics including:
- Execution time
- Relative error
- Precision and recall
- F1 scores
- Confidence intervals

Detailed results and analysis can be found in the paper.


## 🙏 Acknowledgments

We thank the authors of the baseline methods (PQA, SUPG) and the providers of the datasets used in our experiments.

**For questions or issues, please refer to the paper or contact the authors.**

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{wang2026efficient,
  title={On Efficient Approximate Aggregate Nearest Neighbor Queries over Learned Representations},
  author={Wang, Carrie and Amer-Yahia, Sihem and Lakshmanan, Laks and Cheng, Reynold},
  booktitle={ACM SIGMOD 2026},
  year={2026}
}
```
