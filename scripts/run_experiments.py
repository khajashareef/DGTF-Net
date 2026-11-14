"""
Run the full DGTF-Net revision experiment plan (122 runs) on any machine with a CUDA GPU.

  python scripts/run_experiments.py --list               # show all runs and which are finished
  python scripts/run_experiments.py                      # run everything not yet finished
  python scripts/run_experiments.py --worker 0 --n 4     # run every 4th job starting at 0 (split over 4 GPUs)
  python scripts/run_experiments.py --task 17            # run one job by index (Slurm job arrays)
  python scripts/run_experiments.py --only DGTF-Net      # restrict to one model / ablation variant

Finished runs (an existing .json) are skipped, so any command can be re-run after a failure.
Results: one .json + .npz per run under --out (default ./results/dgtf_r1_results).
Run scripts/setup.sh once first (downloads PEMS04/08 and the official baseline code).
"""
import argparse, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DGTF_EXT', os.path.join(ROOT, 'ext'))
os.environ.setdefault('DGTF_DATA', os.path.join(ROOT, 'data'))
sys.path.insert(0, ROOT)
import dgtf_r1 as R  # noqa: E402

DATASETS = ['PEMS08', 'PEMS04']
OWN_BASELINES = ['STGCN', 'DCRNN', 'Graph WaveNet']
MAIN_MODELS = ['DGTF-Net', 'GRU'] + OWN_BASELINES
NEW_MODELS = ['STAEformer', 'STID', 'AGCRN', 'D2STGNN', 'MegaCRN']
SEEDS_MAIN, SEEDS_ABL, SEEDS_NEW = [42, 0, 1, 2, 3], [42, 0, 1], [42, 0, 1]

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'dgtf_r1_results'))
ap.add_argument('--task', type=int)
ap.add_argument('--worker', type=int, default=0)
ap.add_argument('--n', type=int, default=1)
ap.add_argument('--only', default=None, help='model or ablation-variant name')
ap.add_argument('--list', action='store_true')
a = ap.parse_args()

jobs = R.make_jobs(DATASETS, MAIN_MODELS, SEEDS_MAIN, SEEDS_ABL, NEW_MODELS, SEEDS_NEW)
if a.only:
    jobs = [j for j in jobs if (j['variant'] or j['model']) == a.only]
done = lambda j: os.path.exists(R.job_path(a.out, j) + '.json')

if a.list:
    for i, j in enumerate(jobs):
        print(f"{i:3d}  {'DONE' if done(j) else 'todo'}  {j['ds']}  {j['variant'] or j['model']}  seed {j['seed']}")
    print(f'{sum(map(done, jobs))} of {len(jobs)} finished')
    sys.exit(0)

R.persistence_results(a.out, DATASETS)
mine = [jobs[a.task]] if a.task is not None else [j for i, j in enumerate(jobs) if i % a.n == a.worker]
print(f'device: {R.DEVICE} | torch {R.torch.__version__} | {len(mine)} job(s) | out: {a.out}', flush=True)
for j in mine:
    t = time.time()
    R.run_jobs([j], a.out, log=lambda *m: print(*m, flush=True))
    print(f"finished {j['ds']} {j['variant'] or j['model']} seed {j['seed']} in {(time.time() - t) / 60:.1f} min", flush=True)
