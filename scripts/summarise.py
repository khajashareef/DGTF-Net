"""
Rebuild every table, significance test and figure of the revised manuscript from the per-run files.
No GPU and no data download needed.

  python scripts/summarise.py                                   # uses results/dgtf_r1_results
  python scripts/summarise.py --results path/to/dgtf_r1_results
Writes main_table.csv, significance.csv, ablation_table.csv, fuzzy_table.csv, summary.json and the
figures into <results>/_summary.
"""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import dgtf_r1 as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--results', default=os.path.join(ROOT, 'results', 'dgtf_r1_results'))
a = ap.parse_args()
out = os.path.join(a.results, '_summary')
R.summarise(a.results, out)
R.make_figures(a.results, out, models_order=['Persistence', 'GRU', 'STGCN', 'DCRNN', 'Graph WaveNet',
                                             'STAEformer', 'STID', 'AGCRN', 'D2STGNN', 'MegaCRN', 'DGTF-Net'])
print('written to', out)
