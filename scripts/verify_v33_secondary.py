#!/usr/bin/env python3
"""Pure standard-library cross-table verification for ADSv3 v3.3 RC1 S5/S7."""
from collections import defaultdict
from pathlib import Path
import csv
import math

root = Path(__file__).resolve().parents[1]
def records(filename):
    with (root / 'DATA_TABLES' / filename).open(newline='') as f:
        return list(csv.DictReader(f))
def close(a,b,tol=1e-9):
    if not math.isclose(float(a), float(b), abs_tol=tol, rel_tol=0):
        raise AssertionError((a,b))

states = records('S5_AUDITED_144_STATES_20261105.csv')
cohorts = records('S5_AUDITED_72_COHORT_CONTRASTS_20261105.csv')
seeds = records('S5_AUDITED_12_SEED_CONTRASTS_20261105.csv')
s7 = records('S7_TRAINED_VS_PUBLIC_12_PAIRS.csv')
q = records('trained_base_12_checkpoint_effects.csv')
assert (len(states),len(cohorts),len(seeds),len(s7),len(q)) == (144,72,12,12,12)
rs = {}
for r in states:
    k=(r['family'],r['pair'],int(r['cohort']),r['arm'])
    assert k not in rs
    rs[k]=r
    assert int(r['n_images'])==50000
    close(100*int(r['clean_correct'])/50000,r['clean_top1_pct'])
    close(100*int(r['perturbed_correct'])/50000,r['perturbed_top1_pct'])
    close(r['damage_pp'],float(r['clean_top1_pct'])-float(r['perturbed_top1_pct']))
assert len(rs)==144
cmap={}
for r in cohorts:
    k=(r['family'],r['pair'],int(r['cohort']))
    assert k not in cmap
    cmap[k]=r
    rot=rs[k+('ROT180',)]
    flip=rs[k+('HFLIP',)]
    close(r['rot180_damage_pp'],rot['damage_pp'])
    close(r['hflip_damage_pp'],flip['damage_pp'])
    close(r['contrast_rot180_minus_hflip_pp'],float(rot['damage_pp'])-float(flip['damage_pp']))
    assert float(r['contrast_rot180_minus_hflip_pp'])>0
assert len(cmap)==72
by={(r['family'],r['pair']):r for r in seeds}
public={'DeiT':83.774,'Swin':83.754}
for family in ['DeiT','Swin']:
    for i in range(1,7):
        p=f'P{i}'
        vals=[cmap[(family,p,j)] for j in range(1,7)]
        r=by[(family,p)]
        for name,field in [('mean_rot180_damage_pp','rot180_damage_pp'),('mean_hflip_damage_pp','hflip_damage_pp'),('mean_contrast_pp','contrast_rot180_minus_hflip_pp')]:
            close(r[name],sum(float(v[field]) for v in vals)/6)
        assert int(r['positive_cohorts']) == 6
        x=next(z for z in s7 if z['architecture']==family and z['trained_replicate']==p)
        close(x['public_top1_pct'],public[family])
        close(x['trained_top1_pct']-0 if isinstance(x['trained_top1_pct'],float) else float(x['trained_top1_pct']),
              100*int(rs[(family,p,1,'HFLIP')]['clean_correct'])/50000)
        assert x['trained_checkpoint_sha256'] == rs[(family,p,1,'HFLIP')]['checkpoint_sha256']
        close(x['trained_minus_public_pp'],float(x['trained_top1_pct'])-public[family])
        qr=next(y for y in q if y['family']==family and int(y['checkpoint_index'])==i)
        signed=float(qr['q_rot180'])-float(qr['q_hflip'])
        assert (signed>0 if family=='DeiT' else signed<0)
    sd=[float(by[(family,f'P{i}')]['mean_contrast_pp']) for i in range(1,7)]
    print(f'{family} S5 mean contrast pp: {sum(sd)/6:+.9f}, six positive seeds')
    trained=[float(x['trained_top1_pct']) for x in s7 if x['architecture']==family]
    print(f'{family} S7 trained mean Top-1: {sum(trained)/6:.9f} vs public {public[family]:.3f}')
close(sum(float(by[('DeiT',f'P{i}')]['mean_contrast_pp']) for i in range(1,7))/6,.8195)
close(sum(float(by[('Swin',f'P{i}')]['mean_contrast_pp']) for i in range(1,7))/6,.4275555555555555)
print('PASS: 144 S5 states, 72 cohort contrasts, 12 seed contrasts; 12 S7 comparisons and signed q references')
