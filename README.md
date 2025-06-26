# On Aggregation Queries over Predicted Nearest Neighbors

This repository contains the implementation and experimental data for the paper **"On Aggregation Queries over Predicted Nearest Neighbors"**

## Paper

The paper will be available later...

## Problem Statement

Given a query point q and a distance threshold t, we want to compute aggregation functions (e.g., average, variance, sum, min, max, median) over the attributes of points that are true nearest neighbors of q within distance t, using a combination of oracle and proxy model for efficiency.

## Algorithms

This repository implements several algorithms for solving AQNNs:

- **SPRinT-C**: Our approach for count-sensitive AQNNs
- **SPRinT-V**: Our approach for value-sensitive AQNNs
- **PQA-RT/PQA-PT**
- **SUPG-RT/SUPG-PT**
- **TopK**

## Datasets

The experiments use the following datasets:
- **Medical**: eICU, MIMIC-III
- **E-commerce**: Yelp, Electronics
- **Social Media**: Jigsaw

## Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/our-repo.git
   cd your-repo
   ```
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Experiments

To reproduce the experimental results from the paper:

```bash
# Run all experiments
   ./myscript.sh

# Or run individual experiments
python main.py --algo SPRinT-V --agg avg --Fname MIMIC-III
   ```

## Results

The experimental results demonstrate that our proposed algorithms achieve:
- Lower relative error in aggregation estimation
- More efficient computation compared to baselines

Detailed results and analysis can be found in the paper.