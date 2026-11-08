from pathlib import Path
import sys
from statistics import mean
from itertools import combinations, product
from collections import defaultdict
from decimal import Decimal
import zipfile,io,csv,re,json,math
if len(sys.argv) != 2:
 print('Usage: python scripts/verify_sr3_table_cells.py /path/to/extracted/SCIENTIFIC_TEXT_DIR')
 raise SystemExit(2)
P=Path(sys.argv[1]).resolve()
R=Path(__file__).resolve().parents[1]
T=P/'source/supplement'
if not T.is_dir():raise RuntimeError(f'Missing author-supplied LaTeX table sources: {T}')
all_checks=[]
def r(name):return list(csv.DictReader((R/name).open(encoding='utf-8-sig',newline='')))
def num(s):
 s=re.sub(r'\\(?:multicolumn|textbf|mathrm|text)\b','',s)
 s=s.replace('$','').replace('{','').replace('}','').replace('\\','').replace('%','').replace(',','').strip()
 return float(s)
def chk(where,label,reported,actual):
 r=num(reported)
 # Printed numbers are rounded. Accept one half unit of last reported place (+ 1e-8).
 shown=reported.strip().replace('$','').replace('\\','').replace('%','').replace('+','').replace(',','')
 match=re.search(r'-?\d+\.\d+|-?\d+',shown)
 if not match:return
 v=match.group();tol=(10**(-len(v.split('.')[-1])))/2+2e-8 if '.' in v else .50000002
 ok=abs(r-float(actual))<=tol
 all_checks.append({'source_table':where,'item':label,'reported':r,'calculated':float(actual),'match':ok,'tol':tol})
 if not ok:print('MISMATCH',where,label,reported,actual)
def cells(line):
 return [i.strip() for i in line.strip().split(' \\')[0].split('&')]
def lines(filename):
 return [x.strip() for x in (T/filename).read_text().splitlines() if '&' in x and '\\' in x]
# trained contrasts table
x=r('DATA_TABLES/trained_base_288_state_q.csv')
by=defaultdict(dict)
for z in x:by[(z['family'],int(z['checkpoint_index']),int(z['cohort']))][z['operator']]=float(z['q_plusF'])
for l in lines('Table_Training_Seed_Contrasts.tex'):
 m=re.match(r'^(DeiT|Swin) & Seed (\d+) &',l)
 if not m:continue
 c=cells(l);fam=m[1];i=int(m[2]);keys=[by[fam,i,j] for j in range(1,7)];rh=mean(z['HFLIP'] for z in keys);ro=mean(z['ROT180'] for z in keys)
 for label,shown,actual in zip(['q_H','q_R','gap'],c[2:5],[rh,ro,(ro-rh if fam=='DeiT' else rh-ro)]):chk('Table_Training_Seed_Contrasts',f'{fam} P{i} {label}',shown,actual)
# full damage table
ss=r('DATA_TABLES/S5_AUDITED_12_SEED_CONTRASTS_20261105.csv');source={(z['family'],int(z['pair'][1:])):z for z in ss}
for l in lines('Table_Full_Validation_Damage.tex'):
 m=re.match(r'^(DeiT|Swin) & (\d+) &',l)
 if not m:continue
 c=cells(l);fam=m[1];i=int(m[2]);z=source[fam,i]
 for label,shown,k in zip(['Rot','Flip','Diff'],c[2:5],['mean_rot180_damage_pp','mean_hflip_damage_pp','mean_contrast_pp']):chk('Table_Full_Validation_Damage',f'{fam} P{i} {label}',shown,z[k])
# A6 14 models
ss=r('A6_DIRECTIONAL/A6_14_MODEL_MEAN_CONTRASTS.csv');by_model={(z['architecture'],z['model']):z for z in ss}
for l in lines('Table_Image_Disjoint_Directional_Results.tex'):
 m=re.match(r'^(DeiT|Swin) & (Checkpoint (\d+)|Public Base) &',l)
 if not m:continue
 c=cells(l);fam=m[1];model=(fam+'_P'+m[3]) if m[3] else (fam+'_PUBLIC')
 z=by_model[fam,model]
 for label,shown,k in zip(['rot','flip','diff'],c[2:5],['mean_rot180','mean_hflip','mean_rot180_minus_hflip']):chk('Table_Image_Disjoint_Directional_Results',f'{model} {label}',shown,z[k])
# locality
by_cohort={int(z['cohort']):z for z in r('DATA_TABLES/deit_locality_six_cohorts.csv')}
for l in lines('Table_DeiT_Target_Locality.tex'):
 m=re.match(r'^(\d+) &',l)
 if not m:continue
 c=cells(l);z=by_cohort[int(m[1])]
 for label,shown,k in zip(['global','tiled','change'],c[1:4],['global14_rot180_minus_hflip','tiled7_rot180_minus_hflip','tiled7_minus_global14']):chk('Table_DeiT_Target_Locality',f'cohort{m[1]} {label}',shown,z[k])
# clean accuracy
acc={(z['architecture'],int(z['trained_replicate'][1:])):z for z in r('DATA_TABLES/S7_TRAINED_VS_PUBLIC_12_PAIRS.csv')}
for l in lines('Table_Public_Clean_Accuracy.tex'):
 m=re.match(r'^(\d+) &',l)
 if not m:continue
 c=cells(l)
 for fam,offset in [('DeiT',1),('Swin',4)]:
  z=acc[fam,int(m[1])]
  for label,shown,k in zip(['trained','public','diff'],c[offset:offset+3],['trained_top1_pct','public_top1_pct','trained_minus_public_pp']):chk('Table_Public_Clean_Accuracy',f'{fam} P{m[1]} {label}',shown,z[k])
# common beta DeiT 12x
beta={(z['scale'],z['effect']):z for z in r('DATA_TABLES/deit_common_beta_exact_by_scale.csv')}
for l in lines('Table_DeiT_Regularization_Sensitivity.tex'):
 m=re.match(r'^(Small|Base|Large) &',l)
 if not m:continue
 c=cells(l)
 if c[1].startswith('$q'):key='ROT180 axis'
 else:
  key='ROT180--'+('CLEAN' if c[1].endswith('CE') else c[1].split('HFLIP')[-1] if False else c[1].replace('ROT180$-$',''))
 z=beta[m[1],key]
 chk('Table_DeiT_Regularization_Sensitivity',f'{m[1]} {key} mean',c[2],z['mean'])
 chk('Table_DeiT_Regularization_Sensitivity',f'{m[1]} {key} p',c[4],z['nominal_one_sided_p'])
 chk('Table_DeiT_Regularization_Sensitivity',f'{m[1]} {key} L95',c[5],z['one_sided_L95_if_supported'])
# historically nested DeiT old schedule and Swin scale panels
olddeit=r('DATA_TABLES/public_deit_legacy_state_parameter_readouts.csv')
oldswin=r('DATA_TABLES/directional_swin_72_original_q_states.csv')
def locbound(xs):return sorted(mean(t) for k in range(1,7) for t in combinations(xs,k))[2]
def pp(xs):return sum(sum(s*x for s,x in zip(v,xs))>=sum(xs)-1e-14 for v in product((1,-1),repeat=6))/64
for l in lines('Table_Public_Exact_Contrasts.tex'):
 m=re.match(r'^(DeiT|Swin) & (Small|Base|Large|Tiny) &',l)
 if not m:continue
 c=cells(l);fam=m[1];scale=m[2];source=olddeit if fam=='DeiT' else oldswin
 q={(int(z['cohort']),z['arm']):float(z['q_plusF']) for z in source if z['scale']==scale}
 arm='CLEAN_LABEL_FREQS_CONTINUATION' if fam=='DeiT' else 'CLEAN'
 effect=c[2].replace('$','').replace('q(+F)','').strip()
 if effect.startswith('ROT180') and not '-' in effect and not 'HFLIP' in effect:values=[q[i,'ROT180'] for i in range(1,7)]
 else:
  control='ROT90' if 'ROT90' in effect else ('HFLIP' if 'HFLIP' in effect else arm)
  values=[q[i,'ROT180']-q[i,control] for i in range(1,7)]
 chk('Table_Public_Exact_Contrasts',f'{fam} {scale} {effect} mean',c[3],mean(values))
 ppart='1/64' if c[-1].strip().startswith('1/64') else c[-1].strip().split(' / ')[0]
 if ppart=='1.000':chk('Table_Public_Exact_Contrasts',f'{fam} {scale} {effect} p',ppart,pp(values))
 elif ppart=='1/64':chk('Table_Public_Exact_Contrasts',f'{fam} {scale} {effect} p',str(1/64),pp(values))
 else:chk('Table_Public_Exact_Contrasts',f'{fam} {scale} {effect} p',ppart,pp(values))
# full source standardized table: data 72 states/sigma calibrated
nd=r('DATA_TABLES/finite_null_analytic_sigma_186_states.csv')
sigma={(z['model_family'],z['scale'],z['cohort_index'],z['operator']):float(z['analytic_sigma']) for z in nd}
d=r('DATA_TABLES/deit_common_beta_72_state_q.csv');sw=r('DATA_TABLES/directional_swin_72_original_q_states.csv')
for l in lines('Table_Public_Standardized_Alignment.tex'):
 m=re.match(r'^(DeiT|Swin) & (Small|Base|Large|Tiny) &',l)
 if not m:continue
 c=cells(l);fam=m[1];scale=m[2];source=d if fam=='DeiT' else sw
 arr=[z for z in source if z['scale']==scale]
 for i,arm in enumerate(['ROT180','HFLIP','CLEAN'],2):
  z=[]
  for z0 in arr:
   if z0.get('operator',z0.get('arm'))!=arm:continue
   z.append(float(z0['q_z_analytic']) if fam=='DeiT' else float(z0['q_plusF'])/sigma['Swin mixed RoPE',scale,z0['cohort'],arm])
  chk('Table_Public_Standardized_Alignment',f'{fam} {scale} {arm} z',c[i],mean(z))
 # damage of rot - flip from state-level
 q={ (int(z['cohort']),z.get('operator',z.get('arm'))):float(z['damage_pp']) for z in arr }
 damage=mean(q[i,'ROT180']-q[i,'HFLIP'] for i in range(1,7))
 chk('Table_Public_Standardized_Alignment',f'{fam} {scale} damage',c[5],damage)
# secondary summary 5 means
sums={z['measure']:z for z in r('DATA_TABLES/trained_checkpoint_secondary_direction_by_seed.csv')}
for l in lines('Table_Secondary_Checkpoint_Results.tex'):
 m=re.match(r'^(DeiT axis|DeiT contrast|Swin axis|DeiT CE|Swin CE) &',l)
 if not m:continue
 c=cells(l)
 mapping={'DeiT axis':'deit_rot180_axis','DeiT contrast':'deit_rot180_minus_rot90','Swin axis':'swin_hflip_axis','DeiT CE':'deit_clean','Swin CE':'swin_clean'}
 z=sums[mapping[m[1]]]
 chk('Table_Secondary_Checkpoint_Results',f'{m[1]} mean',c[2],z['mean'])
 if z['one_sided_L95']:chk('Table_Secondary_Checkpoint_Results',f'{m[1]} L95',c[5],z['one_sided_L95'])
result={'total_checked_published_numeric_cells':len(all_checks),'passed':sum(x['match'] for x in all_checks),'failed':[x for x in all_checks if not x['match']],'coverage_by_table':{k:{'cells':sum(x['source_table']==k for x in all_checks),'passed':sum(x['source_table']==k and x['match'] for x in all_checks)} for k in sorted(set(x['source_table'] for x in all_checks))},'checks':all_checks}
out=R/'OUTPUT'/'SR3_TABLE_CELL_CROSSCHECK.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result,indent=2)+'\n')
print('SR3_SUPPLEMENT_TABLE_CELLS',len(all_checks),'PASS',sum(x['match'] for x in all_checks),'FAIL',len(result['failed']))
print('SR3_TABLES_COVERED',len(result['coverage_by_table']))
if result['failed']:raise SystemExit(1)
