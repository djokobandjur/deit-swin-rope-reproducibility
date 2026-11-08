#!/usr/bin/env python3
"""Regenerate optional draft scientific figures from the archived numeric CSVs."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent
with (P/'A6_VS_STEP9_12_SEED_COMPARISON.csv').open() as f: d=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(6.4,5.1))
for arch,marker in [('DeiT','o'),('Swin','s')]:
 rows=[r for r in d if r['architecture']==arch]
 x=[float(r['historical_ROT180_minus_HFLIP']) for r in rows];y=[float(r['A6_fresh_ROT180_minus_HFLIP']) for r in rows]
 ax.scatter(x,y,label=arch+' (6 trained checkpoints)',marker=marker,s=48)
lim=(-0.078,0.086)
ax.plot(lim,lim,linestyle='--',linewidth=1.0,color='0.5',label='Equal historical and fresh-image contrast')
ax.set_xlim(*lim);ax.set_ylim(*lim);ax.set_aspect('equal','box')
ax.axhline(0,linewidth=.7,color='0.7');ax.axvline(0,linewidth=.7,color='0.7')
ax.set_xlabel('Historical STEP9 contrast: ROT180 − HFLIP')
ax.set_ylabel('Fresh-image A6 contrast: ROT180 − HFLIP')
ax.legend(fontsize=8,frameon=False,loc='upper left')
fig.tight_layout()
fig.savefig(P/'Figure_A6_Historical_vs_Fresh.pdf')
fig.savefig(P/'Figure_A6_Historical_vs_Fresh.png',dpi=180)
plt.close(fig)

with (P/'A6_SWIN_STAGE_MODEL_MEANS.csv').open() as f: s=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(6.8,4.3))
for group,marker in [('TRAINED','o'),('PUBLIC','s')]:
 rr=[r for r in s if r['source']==group]
 xs=range(1,5)
 yy=[np.mean([float(r['mean_rot180_minus_hflip_stage']) for r in rr if int(r['stage'])==stage]) for stage in xs]
 ax.plot(list(xs),yy,marker=marker,label='Swin '+('trained (six-seed mean)' if group=='TRAINED' else 'public Base'))
ax.axhline(0,color='0.5',linewidth=.8)
ax.set_xticks([1,2,3,4])
ax.set_xlabel('Swin stage (1-based)')
ax.set_ylabel('Contribution to ROT180 − HFLIP global q')
ax.legend(frameon=False,fontsize=8)
fig.tight_layout()
fig.savefig(P/'Figure_A6_Swin_Stage_Contributions.pdf')
fig.savefig(P/'Figure_A6_Swin_Stage_Contributions.png',dpi=180)
plt.close(fig)
