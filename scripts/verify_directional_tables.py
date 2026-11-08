#!/usr/bin/env python3
"""Public numeric verification: archived q values and cohort contrasts only.
No checkpoint/gradient/perturbation replay is claimed by this script.
"""
from pathlib import Path
import csv, itertools
import numpy as np
BASE=Path(__file__).resolve().parents[1]/'DATA_TABLES'
def rows(filename):
    with (BASE/filename).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def exact_one_sided_positive(x):
    x=np.asarray(x,dtype=float)
    observed=x.mean()
    vals=[np.mean(x*np.array(bits)) for bits in itertools.product((-1,1),repeat=len(x))]
    return sum(t >= observed-1e-14 for t in vals)/len(vals)
def exact_lower95(x):
    x=np.asarray(x,dtype=float)
    # Discrete inversion: third smallest mean across nonempty subsets at n=6,
    # for a one-sided alpha=.05 and 64 exhaustive sign assignments.
    subsetmeans=sorted(np.mean(x[list(idx)]) for n in range(1,len(x)+1)
      for idx in itertools.combinations(range(len(x)),n))
    return subsetmeans[2]

def contrast(records,order,grp):
    out=[]
    groups={}
    for r in records:
        k=tuple(r[key] for key in grp)
        groups.setdefault(k,{})[r['operator']]=float(r['q_plusF'])
    for k,arms in groups.items():
        assert set(order).issubset(arms), (k,arms)
        out.append((k,arms[order[0]]-arms[order[1]]))
    return out

de=rows('deit_common_beta_72_state_q.csv')
sw=rows('directional_swin_72_original_q_states.csv')
for r in sw:r['operator']=r['arm']
assert len(de)==len(sw)==72
for family,data,ops in [('DeiT',de,('ROT180','HFLIP')),('Swin',sw,('HFLIP','ROT180'))]:
    pair=contrast(data,ops,('scale','cohort'))
    assert len(pair)==18 and all(v>0 for _,v in pair)
    cohorts=[np.mean([v for (scale,cohort),v in pair if cohort==str(i)]) for i in range(1,7)]
    print(f'{family} fixed public: mean {np.mean(cohorts):.12f}, positive 18/18, p={exact_one_sided_positive(cohorts):.6f}, L95={exact_lower95(cohorts):.9f}')
loc=rows('deit_locality_six_cohorts.csv');assert len(loc)==6
G=np.array([float(r['global14_rot180_minus_hflip']) for r in loc]);L=np.array([float(r['tiled7_rot180_minus_hflip']) for r in loc]);delta=L-G
assert np.all(delta<0) and np.all(L>0)
print(f'DeiT locality: global {G.mean():.12f}, tiled {L.mean():.12f}, change {delta.mean():.12f}, reduction {100*(1-L.mean()/G.mean()):.5f}%')
trained=rows('trained_base_288_state_q.csv');assert len(trained)==288
assert len({(r['family'],r['checkpoint_index'],r['cohort'],r['operator']) for r in trained})==288
summary=rows('trained_base_12_checkpoint_effects.csv');assert len(summary)==12
for family,ops,target_mean,target_bound in [('DeiT',('ROT180','HFLIP'),0.04958352486032997,0.043087615263774845),('Swin',('HFLIP','ROT180'),0.04279591677032235,0.03200899193858969)]:
    group=[r for r in trained if r['family']==family]
    contrasts=contrast(group,ops,('checkpoint_index','cohort'))
    assert len(contrasts)==36 and all(v>0 for _,v in contrasts)
    seeds=np.array([np.mean([v for (seed,cohort),v in contrasts if seed==str(i)]) for i in range(1,7)])
    assert abs(seeds.mean()-target_mean)<1e-12
    assert abs(exact_lower95(seeds)-target_bound)<1e-12
    assert exact_one_sided_positive(seeds)==1/64
    print(f'{family} new trained Base: mean {seeds.mean():.12f}, positive 6/6 seeds and 36/36 cohorts, p={exact_one_sided_positive(seeds):.6f}, L95={exact_lower95(seeds):.12f}')
print('PASS: all public numeric checks')
