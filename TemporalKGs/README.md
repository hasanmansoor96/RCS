# TemporalKGs Signal Construction Pipeline

## Overview
This folder contains `all_signals.ipynb`, an end-to-end notebook that builds three pairwise signals from ICEWS event data:

1. Co-occurrence influence (`sparse_influence_matrix.json`)
2. Granger-style temporal influence (`granger_influence_matrix_optimized.json`)
3. Structural similarity (`emb_sim_matrix.json`)

The notebook is the latest integrated workflow and includes preprocessing, filtering, optimization, and export.

## Main Input
- `icews05-15_aug_inverse_time_year/icews_2005-2015_train_normalized.txt`

The notebook loads this file, normalizes the time axis, and creates fixed time windows.

## What `all_signals.ipynb` Does

### 1. Data preparation
- Loads ICEWS triples/events.
- Renames columns to a consistent schema (`time`, `subject_id`, `object_id`).
- Converts dates to integer day offsets from the minimum date.

### 2. Time windowing
- Uses window size `W = 30` days.
- Assigns each event a `window_id`.

### 3. Co-occurrence signal
- For each window, builds a set of active entities.
- Counts ordered co-occurrence pairs `(A, C)` across windows.
- Computes normalized score:
  - `normalized(A,C) = count(A,C) / (count_win(A) * count_win(C))`
- Applies min-max scaling to `[0,1]`.
- Applies sparsity thresholds:
  - `min_raw_count = 3`
  - `min_scaled_score = 0.01`
- Saves sparse matrix as nested JSON:
  - `A -> {C: scaled_score}`

### 4. Granger-style temporal influence (optimized)
- Builds `entity_activity_df` (entity-by-window activity counts).
- Starts from sparse co-occurrence candidates, then further filters:
  - top `30` neighbors per source entity
  - raw overlap count `>= 5`
- Fits two OLS models per pair `(A,C)`:
  - Baseline: predict `C_t` from lagged `C`
  - Full: predict `C_t` from lagged `C` and lagged `A`
- Influence score:
  - `delta_R2 = max(0, R2_full - R2_baseline)`
- Uses `p = 1` lag and `min_time_points = p + 2`.
- Min-max scales scores and saves:
  - `granger_influence_matrix_optimized.json`

### 5. Structural similarity signal
- Builds per-entity structural vectors over `(relation, role, window)` features.
- Uses cosine similarity for filtered pairs (same top-k/overlap strategy).
- Clips negative similarity to `0`.
- Two-pass scaling strategy:
  - Pass 1: find global min/max clipped cosine score.
  - Pass 2: scale and incrementally write JSON to reduce memory pressure.
- Saves:
  - `emb_sim_matrix.json`

## Output Files
- `sparse_influence_matrix.json`: Sparse co-occurrence influence graph.
- `granger_influence_matrix_optimized.json`: Directed temporal influence scores.
- `emb_sim_matrix.json`: Directed pairwise structural similarity scores.

All outputs are nested dictionaries in JSON:
- outer key: source entity `A`
- inner key: target entity `C`
- value: normalized score in `[0,1]`

## Run Statistics Captured in Notebook
- Unique ordered co-occurring pairs: `28,466,426`
- Sparse co-occurrence entries after thresholding: `2,269,914`
- Entity activity matrix shape: `10,094 x 134` (entities x windows)
- Optimized pair count for Granger/structural processing: `13,132`

## Notes and Caveats
- The notebook is iterative and contains repeated setup blocks from reruns.
- Paths are Colab-style in cells (e.g., `/content/...`), but outputs here are local JSON files.
- `subject_country_code` / `object_country_code` are identity-mapped from actor IDs in this workflow.

## Suggested Report Structure
1. Problem statement and motivation
2. Dataset and preprocessing
3. Signal definitions and formulas
4. Filtering/optimization strategy
5. Final artifacts and their interpretation
6. Limitations and next improvements
