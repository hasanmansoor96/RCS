## 1) Project Context

Goal: produce a directed influence graph `A -> C` by combining three signals:
1. Temporal co-occurrence influence
2. Temporal Granger-style dependency
3. Embedding/structural similarity

Final artifact generated so far: `TemporalKGs/final_influence_graph.json`.

## 2) Data Used
### Primary normalized dataset files
- `TemporalKGs/icews05-15_aug_inverse_time_year/icews_2005-2015_train_normalized.txt`
- `TemporalKGs/icews05-15_aug_inverse_time_year/icews_2005-2015_valid_normalized.txt`
- `TemporalKGs/icews05-15_aug_inverse_time_year/icews_2005-2015_test_normalized.txt`

### Normalized schema
TSV columns:
`head, relation, tail, date, year, month, day, time_index, head_country, tail_country, is_domestic`

- `time_index = (year - 2005) * 12 + (month - 1)`
- `is_domestic = 1` when both extracted countries match (case-insensitive), else `0`

### Important implementation note
Current implemented notebooks (`all_signals.ipynb`, `influence_graph.ipynb`) use the **train normalized file** for signal generation. The train/valid/test evaluation protocol is planned but not yet executed end-to-end.

## 3) Initial Plan (What We Intended)
Planned scoring form:

`W(A->C) = alpha * coOcc + beta * dep + gamma * embSim`

Planned experiments:
- all entities
- international-only (`is_domestic=0`)
- domestic-only (`is_domestic=1`)

Planned evaluation:
- future-link prediction with Hits@3/5/10
- monthly first, then weekly/daily variants

## 4) Implemented Pipeline (What Was Actually Built)
Two notebooks are relevant:
- `TemporalKGs/all_signals.ipynb` -> builds the 3 component signals
- `TemporalKGs/influence_graph.ipynb` -> aligns, normalizes, fuses, sparsifies into final graph

### 4.1 Signal A: Co-occurrence (`sparse_influence_matrix.json`)
Implemented in `all_signals.ipynb`:
- Load train normalized events
- Convert date to day offset
- Fixed time windows: `W = 30` days
- For each window, collect active entities
- Count directed pair co-activity `(A, C)` across windows
- Normalize:
  - `coOcc(A,C) = coocc_count(A,C) / (count_win(A) * count_win(C))`
- Min-max scale to `[0,1]`
- Sparsify with thresholds:
  - `min_raw_count = 3`
  - `min_scaled_score = 0.01`

Output:
- `TemporalKGs/sparse_influence_matrix.json`

### 4.2 Signal B: Granger-style Dependency (`granger_influence_matrix_optimized.json`)
Implemented in `all_signals.ipynb`:
- Build entity activity time series by 30-day windows
- Candidate filtering (compute constraint step):
  - top `30` neighbors per source from co-occurrence
  - retain only pairs with raw co-occurrence `>= 5`
- Granger-style score for each pair `(A,C)`:
  - baseline model predicts `C_t` from lagged `C`
  - full model predicts `C_t` from lagged `C` + lagged `A`
  - score = `max(0, R2_full - R2_baseline)`
- Parameters:
  - lag `p = 1`
  - `min_time_points = 3`
- Min-max scale to `[0,1]`

Output:
- `TemporalKGs/granger_influence_matrix_optimized.json`

### 4.3 Signal C: Embedding/Structural Similarity (`emb_sim_matrix.json`)
Implemented in `all_signals.ipynb`:
- Build per-entity structural feature vectors over `(relation, role, window)`
- Use cosine similarity for filtered candidate pairs
- Clip negative cosine to `0`
- Two-pass min-max scaling (memory-optimized)
- Incremental JSON writing to avoid keeping full dense output in memory
- Uses same filtered pair strategy as Granger:
  - top `30`
  - raw co-occurrence `>= 5`

Output:
- `TemporalKGs/emb_sim_matrix.json`

### 4.4 Fusion into Final Graph (`final_influence_graph.json`)
Implemented in `influence_graph.ipynb`:
- Load three JSON signals:
  - `sparse_influence_matrix.json`
  - `granger_influence_matrix_optimized.json`
  - `emb_sim_matrix.json`
- Build a master entity index from all sources/targets
- Align all matrices to the master index (missing values -> `0`)
- Row-wise min-max normalize each aligned matrix
- Fuse with fixed weighted sum:
  - co-occurrence: `0.4`
  - granger: `0.4`
  - embedding similarity: `0.2`
- Sparsify fused graph:
  - keep top `30` neighbors per source
  - apply minimum fused weight threshold `0.05`

Output:
- `TemporalKGs/final_influence_graph.json`

## 5) Key Decisions and Clarifications
- **Top-k is implemented and fixed at 30** (not open anymore).
- This top-k filtering was introduced due to computational constraints.
- Fusion currently uses **fixed coefficients (0.4/0.4/0.2)**, not learned `alpha/beta/gamma`.
- Current implemented signal generation is train-only.
- Final output is sparsified and normalized through the described pipeline.

## 6) What Was Planned But Is Not Yet Completed
1. Learn `alpha/beta/gamma` from a future-link objective (instead of fixed weights).
2. Run formal temporal evaluation protocol with Hits@3, Hits@5, Hits@10.
3. Execute full split-aware experiments (train/valid/test) for future-link prediction.
4. Run segmentation experiments:
   - all entities
   - international-only (`is_domestic=0`)
   - domestic-only (`is_domestic=1`)
5. Compare monthly vs weekly vs daily time-series definitions.
6. Produce final graph visualizations and analysis package.

## 7) Challenges and Blockers Encountered
- Full pairwise processing was too expensive at ICEWS scale.
- Needed aggressive candidate pruning (`top_k=30`, raw overlap thresholding) for tractability.
- Notebook development was iterative; repeated setup cells appear due to reruns.
- Colab-style paths (`/content/...`) required care when running locally.
- Memory pressure required two-pass and incremental writing for embedding similarity.

## 8) Current Artifacts
- `TemporalKGs/sparse_influence_matrix.json`
- `TemporalKGs/granger_influence_matrix_optimized.json`
- `TemporalKGs/emb_sim_matrix.json`
- `TemporalKGs/final_influence_graph.json`
- `TemporalKGs/all_signals.ipynb`
- `TemporalKGs/influence_graph.ipynb`

## 9) Recommended Next Steps
1. Lock an explicit temporal split for future-link evaluation (inside train and/or train->valid/test).
2. Define positive/negative construction for ranking candidates per source entity.
3. Learn combination weights (`alpha,beta,gamma`) against Hits@K objective.
4. Re-run fusion with learned weights and compare against fixed 0.4/0.4/0.2 baseline.
5. Run international vs domestic variants and document differences.
6. Add visualization notebook/scripts (top influencers, subgraphs, temporal snapshots).

## 10) Repro Notes
- Generate component signals via `TemporalKGs/all_signals.ipynb`.
- Generate final fused graph via `TemporalKGs/influence_graph.ipynb`.
- If running locally, update notebook paths from `/content/...` to local repository paths.
