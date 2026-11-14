# DGTF-Net

Code and results for the article

> **Gated Fusion of Adaptive Graph Attention and Temporal Transformers for Accurate Urban Traffic Congestion Prediction**
> Sk Khaja Shareef, N. V. Raja Sekhar Reddy, Mallareddy Adudhodla, Durga Prasad Kavadi, Kurumalla Suresh, Khadija Jabeen.
> *Discover Computing* (under revision).

DGTF-Net (Dual-Graph Transformer Fusion Network) forecasts traffic flow 12 steps (60 min) ahead from the past 12 steps. It combines
a graph-biased multi-head attention over a gated fusion of a prior road graph and an adaptive graph, a per-node temporal
Transformer, element-wise gated fusion of the two branches, and a post-processing fuzzy stage that turns forecasts into
Low / Medium / High congestion severity.

Every number, table and figure in the revised manuscript is produced by the code in this repository, and the raw per-run
results are included.

## Results (test set, average over all 12 steps, mean ± std over seeds)

| Model | PEMS08 MAE | PEMS08 RMSE | PEMS04 MAE | PEMS04 RMSE | Params (PEMS08) |
|---|---|---|---|---|---|
| Graph WaveNet ‡ | 15.19 ± 0.07 | 24.17 ± 0.11 | 19.72 ± 0.05 | 31.39 ± 0.07 | 0.31 M |
| STID ‡ | 14.25 ± 0.01 | 23.24 ± 0.01 | 18.69 ± 0.03 | 30.29 ± 0.03 | 0.12 M |
| STAEformer ‡ | 13.54 ± 0.00 | 23.36 ± 0.05 | 18.35 ± 0.09 | 30.43 ± 0.25 | 1.22 M |
| **DGTF-Net** | **14.00 ± 0.03** | **23.17 ± 0.03** | **18.74 ± 0.02** | **30.36 ± 0.03** | 0.36 M |

‡ = existing implementation retrained in our training loop. The full tables with all ten baselines, per-horizon metrics,
significance tests, the ablation and the fuzzy-stage evaluation are in `results/dgtf_r1_results/_summary/`.

## Repository layout

```
dgtf_r1.py                         all code: data, DGTF-Net (with ablation switches), baseline wrappers,
                                   training loop, metrics, fuzzy stage, job queue, statistics, figures
notebooks/DGTF-Net_R1_Experiments.ipynb   the Google Colab notebook used for the runs (same code)
scripts/setup.sh                   one-time setup: packages, PEMS04/08 data, baseline code at pinned commits
scripts/run_experiments.py         run all 122 runs (or a subset) on any CUDA machine; resumable
scripts/slurm_array.sbatch         the same as a Slurm job array (one run per GPU)
scripts/summarise.py               rebuild all tables, tests and figures from the per-run files (CPU only)
results/dgtf_r1_results/           one .json (metrics, fuzzy results, gate g, timing) + one .npz
                                   (per-window test errors) per run; _summary/ holds the tables and figures
```

## Reproduce the tables and figures (no GPU)

```bash
pip install -r requirements.txt
python scripts/summarise.py
```

This regenerates `main_table.csv`, `significance.csv` (Welch, Mann–Whitney and Diebold–Mariano tests, Holm-corrected),
`ablation_table.csv` (paired t-tests), `fuzzy_table.csv`, `summary.json` and the figures in
`results/dgtf_r1_results/_summary/` from the raw runs.

## Re-run the experiments (GPU)

```bash
bash scripts/setup.sh                                  # needs internet once
python scripts/run_experiments.py --list               # 122 runs and their status
python scripts/run_experiments.py                      # run everything not yet finished
python scripts/run_experiments.py --worker 0 --n 4     # split over 4 GPUs (worker 0..3)
python scripts/run_experiments.py --only DGTF-Net      # one model or ablation variant only
sbatch scripts/slurm_array.sbatch                      # or: Slurm job array, one run per GPU
```

To re-run into a fresh folder, add `--out my_results`. Finished runs are skipped, so any command can be repeated after an
interruption. On Google Colab, open the notebook instead and run the cells in order (results go to Google Drive).

**Run plan.** DGTF-Net, GRU, STGCN, DCRNN and Graph WaveNet: seeds {42, 0, 1, 2, 3}. STAEformer, STID, AGCRN, D²STGNN,
MegaCRN and the seven ablation variants: seeds {42, 0, 1}. Both datasets. Plus the deterministic last-value persistence baseline.

**Shared protocol (all models).** 60/20/20 chronological split; global Z-score normalisation from the training period;
P = Q = 12; Huber loss (δ = 1) on de-normalised flow; Adam (lr 1e-3, weight decay 1e-5); cosine schedule; 30 epochs;
early stopping (patience 12) on validation MAE; gradient clipping 5; batch size 32. Metrics exclude zero readings and
average over all 12 output steps.

**DGTF-Net hyperparameters.** d = 64, 3 blocks, 8 heads, node-embedding size 10, top-k = 20, no dropout.

**Hardware used.** NVIDIA Tesla T4 (16 GB), PyTorch 2.14.1, CUDA 13.0. Seconds per epoch on a T4: DGTF-Net 101 (PEMS08) /
226 (PEMS04).

## Baselines

The baseline code is **not** copied into this repository. `scripts/setup.sh` clones each official repository at the
commit listed in `OFFICIAL_REPOS` in `dgtf_r1.py`, and `dgtf_r1.py` wraps the model classes into the shared training loop:

| Model | Source |
|---|---|
| STGCN, DCRNN, Graph WaveNet | BasicTS benchmark, commit `c218c07` |
| STAEformer | github.com/XDZhelheim/STAEformer, commit `fc49d39` |
| STID | github.com/zezhishao/STID, commit `e8b313b` |
| AGCRN | github.com/LeiBAI/AGCRN, commit `7fbbf2a` |
| D²STGNN | github.com/zezhishao/D2STGNN, commit `82c2d38` |
| MegaCRN | github.com/deepkashiwa20/MegaCRN, commit `5158190` |

PDFormer is not retrained (its official code depends on the LibCity framework); the manuscript lists its published results
for reference only.

## Data

PEMS04 and PEMS08 are public benchmark datasets. `scripts/setup.sh` downloads them from the ASTGNN repository
(github.com/guoshnBJTU/ASTGNN) into `data/`.

## Citation

See `CITATION.cff`. Please cite the article once published, and this software release via its Zenodo DOI.

## License

MIT (see `LICENSE`). The baseline implementations remain under their own licences.
