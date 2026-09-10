# Predicting Tennis Match Outcomes from Pre-Match Playing Style

A machine learning pipeline that predicts ATP match winners using only
**pre-match** statistics  no data from the match itself leaks into the
prediction. Built on the [Jeff Sackmann Tennis Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject)
dataset.

## What this project does

1. **Exploratory analysis** of serve/return statistics across surfaces and
   match outcomes (ace rate, first-serve win %, break-point save rate, etc.)
2. **Unsupervised profiling**: PCA + K-means clustering of match performance
   profiles into distinct playing-style clusters (validated with the
   elbow method and silhouette analysis, K=3)
3. **Supervised prediction**: rolling 20-match pre-match averages per player,
   turned into symmetrised win/loss matchup differentials, with a strict
   *temporal* train/test split (no random shuffling, the model is only
   ever evaluated on matches that happened after its training window)
4. **Model comparison**: logistic regression, elastic net, and random
   forest, evaluated on accuracy, AUC-ROC, F1, Kappa, and Brier score

## Key design choices worth noting

- **No look-ahead leakage**: every predictive feature is a *rolling average
  of a player's past N matches*, computed only from matches strictly before
  the one being predicted.
- **Temporal split, not random**: train/test is split at a cutoff date
  (75th percentile), because a random split would let the model "see the
  future" relative to some training examples.
- **Symmetrised matchups**: each match appears twice, once as
  (winner − loser) labelled Win, once as (loser − winner) labelled Loss,
  so the model can't learn a trivial ordering shortcut.
- **Preprocessing fit on training data only**: centering/scaling
  parameters are learned from the training set and applied to the test
  set, not the reverse.

## Results

| Model | Accuracy | AUC-ROC | F1 | Kappa | Brier |
|---|---|---|---|---|---|
| Logistic Regression | 0.698 | **0.775** | 0.698 | 0.396 | 0.193 |
| Elastic Net | 0.697 | 0.774 | 0.697 | 0.394 | 0.194 |
| Random Forest | 0.685 | 0.758 | 0.683 | 0.370 | 0.200 |

Logistic regression edges out the other two, but the gap between all three
is small enough that model choice shouldn't rest on accuracy alone, it's
also the most interpretable of the three, since coefficients read directly
as log-odds. 5-fold CV ROC (0.782) tracked closely with test-set AUC
(0.775), indicating the model generalises rather than overfits.

**The single strongest predictor by a wide margin is the rolling win-rate
differential** (`d_win_rate`) recent form dominates over any individual
serve/return statistic. This is consistent with the ~70% accuracy ceiling
reported elsewhere in tennis outcome prediction (Kovalchik, 2016); the
model is a useful decision-support baseline rather than a high-confidence
forecaster, since it omits surface, head-to-head history, ranking, and
in-match dynamics.

Clustering identified **3 playing-style clusters** (silhouette-validated,
K=3), interpretable roughly as: consistent/low-error players, error-prone
players, and high-risk/high-reward players who hit more winners but also
make more mistakes. Separation runs along an attacking-play/error-rate
axis (PC1) and a serve-risk axis (PC2), though the clusters show some
overlap, PC1+PC2 explain only ~50% of total variance. See
`output/figures/` for cluster plots and player footprint plots (Federer,
Nadal, Cilic career trajectories in PCA space).

## Limitations

- No surface, head-to-head, or ranking data included as predictors
- In-match statistics contribute little once recent win-rate is accounted for — the model can't capture live match dynamics (correctly excluded here to avoid leakage)
- Can't quantify player fitness, injury status, or psychological factors
- Grand Slam matches (best-of-5) are mixed in with regular tour matches (best-of-3), which may distort raw-count-based features
- ~70% accuracy is consistent with the literature's ceiling for pre-match-only prediction, this is a decision-support baseline, not a high-confidence forecaster

## Repository structure

```
├── R/
│   ├── 01_eda.R                 # Exploratory analysis 
│   ├── 02_clustering.R          # PCA + K-means profiling
│   ├── 03_predictive_model.R    # Rolling features + logistic regression 
│   └── 04_model_comparison.R    # Logistic vs elastic net vs random forest
├── output/
│   ├── figures/                 # All generated plots
│   └── tables/                  # Metrics, coefficients, cluster summaries
├── renv.lock                    # Locked package versions
└── README.md
```

## Data

This project uses the [Tennis Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject)
by Jeff Sackmann, specifically `charting-m-matches.csv` and
`charting-m-stats-overview.csv`. The data is **not included in this repo**
— check the source repository's license before redistributing it, and
download it directly from there into the project root before running
the scripts.

## Reproducing this analysis

```r
# Restore exact package versions
renv::restore()

# Run in order — each script is self-contained and re-derives
# the cleaned dataset from the raw CSVs
source("R/01_eda.R")
source("R/02_clustering.R")
source("R/03_predictive_model.R")
source("R/04_model_comparison.R")
```

Each script sets `set.seed(42)` and writes a `session_info.txt` capturing
the exact R version and package versions used, for full reproducibility.

## Tech stack

R, tidyverse (`dplyr`, `readr`, `tidyr`, `ggplot2`), `caret`, `glmnet`,
`randomForest`, `factoextra`, `pROC`, `cluster`.
