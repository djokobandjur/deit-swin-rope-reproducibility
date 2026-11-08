#!/usr/bin/env python3
"""Compare A6 new-image model means with historical STEP9 model means.
Uses exact historical predeclared 12-model CSV; retrospective comparison only.
"""
import csv, pathlib, statistics, sys, hashlib
p=pathlib.Path(__file__).parent
old=list(csv.DictReader((p/'STEP9_LOCKED_12_MODEL_HISTORICAL_CONTRASTS.csv').open()))
new=list(csv.DictReader((p/'A6_14_MODEL_MEAN_CONTRASTS.csv').open()))
by={r['model']:r for r in new}
assert len(old)==12 and len(by)==14
rows=[]
for r in old:
    mid=f"{r['family']}_P{int(r['checkpoint_index'])}"
    assert mid in by and by[mid]['source']=='TRAINED'
    # Historical table's Swin preferred_contrast is HFLIP minus ROT180;
    # convert to a common ROT180-minus-HFLIP sign convention.
    old_s=(1 if r['family']=='DeiT' else -1)*float(r['preferred_contrast'])
    new_s=float(by[mid]['mean_rot180_minus_hflip'])
    assert old_s*new_s>0
    rows.append({'model':mid,'architecture':r['family'],
                 'historical_ROT180_minus_HFLIP':old_s,
                 'A6_fresh_ROT180_minus_HFLIP':new_s,
                 'A6_minus_historical':new_s-old_s,
                 'historical_positive_cohorts_preferred':int(r['positive_cohorts']),
                 'A6_positive_cohorts_preferred':(int(by[mid]['positive_cohorts']) if r['family']=='DeiT' else int(by[mid]['n_cohorts'])-int(by[mid]['positive_cohorts']))})
with (p/'A6_VS_STEP9_12_SEED_COMPARISON.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
for arch in ['DeiT','Swin']:
    rr=[r for r in rows if r['architecture']==arch]
    print(arch,'history',statistics.mean(r['historical_ROT180_minus_HFLIP'] for r in rr),'A6',statistics.mean(r['A6_fresh_ROT180_minus_HFLIP'] for r in rr),'shift',statistics.mean(r['A6_minus_historical'] for r in rr),'same_sign',sum(r['A6_fresh_ROT180_minus_HFLIP']*r['historical_ROT180_minus_HFLIP']>0 for r in rr),'of',len(rr))
