#!/bin/bash
# One-time setup (needs internet): Python packages, PEMS04/PEMS08 data and the official
# baseline code at the pinned commits listed in dgtf_r1.py (OFFICIAL_REPOS).
set -e
cd "$(dirname "$0")/.."
export DGTF_EXT="$PWD/ext" DGTF_DATA="$PWD/data"
python -m pip install -q -r requirements.txt
python - <<'PY'
import dgtf_r1 as R
R.clone_official()
for ds in ('PEMS04', 'PEMS08'):
    R._fetch_pems(ds)
print('baseline code:', R.EXT_DIR, '| data:', R.DATA_DIR)
PY
python scripts/run_experiments.py --list | tail -1
echo "Setup OK."
