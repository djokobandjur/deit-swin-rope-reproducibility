#!/usr/bin/env python3
"""Recalculate archival numerical results, including A6 F-vector/perturbation readout (CPU)."""
from pathlib import Path
import subprocess,sys
R=Path(__file__).resolve().parents[1]
def run(*args):
 print('RUN', ' '.join(map(str,args)),flush=True)
 subprocess.run([str(x) for x in args],cwd=R,check=True)
run(sys.executable,'scripts/verify_package.py')
run(sys.executable,'scripts/verify_v33_raw.py')
# The historical source-code ZIP is excluded from this public archive.
# Its required original numerical data are supplied under DATA_TABLES/.
# The current independent verifiers below check the reproduced values.
# Re-run numerical verifiers on the bundled table-only data.
run(sys.executable,'scripts/verify_directional_tables.py')
run(sys.executable,'scripts/verify_v33_secondary.py')
run(sys.executable,'A6_DIRECTIONAL/compare_historical.py')
out=R/'OUTPUT'/'A6_DIRECTIONAL_REPLAY'
assert not out.exists(),'OUTPUT_A6_EXISTS: remove OUTPUT after inspecting before rerun'
run(sys.executable,'A6_DIRECTIONAL/recompute_a6_f.py',
 '--fzip','A6_DIRECTIONAL/ADSV3_14_F_VECTORS_VERIFIED_20261106.zip',
 '--a6zip','ARCHIVES/a6/A6_ORIGINAL_168_RAW.zip','--out',out)
run(sys.executable,'scripts/compare_a6_replay.py','A6_DIRECTIONAL',out)
run(sys.executable,'scripts/verify_sr3_research_claims.py')
print('ADSV3_V33_FINAL_ARCHIVED_CPU_REPRODUCTION_PASS')
