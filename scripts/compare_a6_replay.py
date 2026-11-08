#!/usr/bin/env python3
"""Compare fresh torch-backed A6 reconstruction to shipped scientific source tables."""
import csv, math, json, sys
from pathlib import Path

original=Path(sys.argv[1]); reproduced=Path(sys.argv[2]);

KEYS={
  'A6_168_STATE_Q_XY.csv':['model','cohort','arm'],
  'A6_84_PAIRED_Q_CONTRASTS.csv':['model','cohort'],
  'A6_14_MODEL_MEAN_CONTRASTS.csv':['model'],
  'A6_SWIN_336_STAGE_STATE_CONTRIBUTIONS.csv':['model','cohort','arm','stage'],
  'A6_SWIN_168_STAGE_PAIR_CONTRASTS.csv':['model','cohort','stage'],
  'A6_SWIN_STAGE_MODEL_MEANS.csv':['model','stage'],
}

def number(v):
    try: return float(v)
    except (ValueError,TypeError): return None

for filename,keys in KEYS.items():
    with (original/filename).open(newline='') as f: a=list(csv.DictReader(f))
    with (reproduced/filename).open(newline='') as f: b=list(csv.DictReader(f))
    def index(rows):
        return {tuple(r[k] for k in keys):r for r in rows}
    x=index(a); y=index(b)
    assert len(x)==len(a)==len(b)==len(y),filename+' duplicate keys'
    assert x.keys()==y.keys(),filename+' keys'
    for k in x:
        assert x[k].keys()==y[k].keys(),(filename,k,'columns')
        for field,ref in x[k].items():
            candidate=y[k][field]
            p,q=number(ref),number(candidate)
            if p is not None and q is not None:
                assert math.isclose(p,q,rel_tol=0,abs_tol=2e-11),(filename,k,field,ref,candidate)
            else:
                assert ref==candidate,(filename,k,field,ref,candidate)
    print('A6_REPLAY_TABLE_PASS',filename,len(a),flush=True)

ar=json.loads((original/'A6_DIRECTIONAL_RESULTS.json').read_text())
br=json.loads((reproduced/'A6_DIRECTIONAL_RESULTS.json').read_text()); assert ar['a6_fresh_class_count_recomputed']==777 and br['a6_fresh_class_count_recomputed'] is None, 'CLASS_COUNT_REDACTION_BOUNDARY'
for k in ['state_n','pair_n','source_payload_files_sha_verified','F_vectors_checked','a6_fresh_images','fresh_image_overlap_with_historical']:
    assert ar[k]==br[k],(k,ar[k],br[k])
for arch in ['DeiT','Swin']:
    for k in ['mean_q_ROT180_minus_HFLIP','mean_x_contribution','mean_y_contribution','sd_seed','exploratory_one_sided_sign_flip_p']:
        assert math.isclose(ar['trained'][arch][k],br['trained'][arch][k],rel_tol=0,abs_tol=2e-11),(arch,k)
print('ADSV3_SR2_A6_FRESH_REPLAY_PASS',flush=True)
