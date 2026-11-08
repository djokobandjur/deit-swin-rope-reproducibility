#!/usr/bin/env python3
"""Verify a bounded, explicit inventory of SR3 numerical claims using packaged V33 CSVs.

This audit NEVER reads manuscript values as ground truth: published target values
are embedded below, derived values come from distributed data files.
It does not regenerate GPU inference or examine journal PDF layout.
"""
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from statistics import mean,stdev,median
import csv,json,math,sys
from scipy.stats import kstest,chi2

R=Path(__file__).resolve().parents[1]
checks=[]
def rows(path):
 with (R/path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def ok(name,got,expected,tol=0.0000006,source=''):
 good=abs(float(got)-float(expected))<=tol
 checks.append({'name':name,'status':'PASS' if good else 'FAIL','computed':float(got),'paper':float(expected),'tolerance':tol,'source':source})
 if not good:print('MISMATCH',name,got,expected,file=sys.stderr)
 return good
def count(name,got,expected,source=''):
 checks.append({'name':name,'status':'PASS' if got==expected else 'FAIL','computed':int(got),'paper':int(expected),'tolerance':0,'source':source})
def location_bound(xs):
 return sorted(mean(c) for size in range(1,len(xs)+1) for c in combinations(xs,size))[2]
def p_positive(xs):
 return sum(mean([s*x for s,x in zip(signs,xs)])>=mean(xs)-1e-14 for signs in __import__('itertools').product((1,-1),repeat=len(xs)))/2**len(xs)

deit=rows('DATA_TABLES/deit_common_beta_72_state_q.csv')
swin=rows('DATA_TABLES/directional_swin_72_original_q_states.csv')
null=rows('DATA_TABLES/finite_null_analytic_sigma_186_states.csv')
source='DATA_TABLES/deit_common_beta_72_state_q.csv; DATA_TABLES/directional_swin_72_original_q_states.csv'
scale_gaps={}
for fam,rr,scales in [('DeiT',deit,['Small','Base','Large']),('Swin',swin,['Tiny','Small','Base'])]:
 for scale in scales:
  q={(int(x['cohort']),x.get('operator',x.get('arm'))):float(x['q_plusF']) for x in rr if x['scale']==scale}
  sign=1 if fam=='DeiT' else -1
  gaps=[sign*(q[i,'ROT180']-q[i,'HFLIP']) for i in range(1,7)]
  count(f'{fam} {scale} per-cohort preferred signs',sum(x>0 for x in gaps),6,source)
  scale_gaps[(fam,scale)]=gaps
  for (f,s,v) in [('DeiT','Small',.02377929849),('DeiT','Base',.04926575731),('DeiT','Large',.04276049339),('Swin','Tiny',.01990261314),('Swin','Small',.02336035211),('Swin','Base',.03290474571)]:
   if (f,s)==(fam,scale):ok(f'{fam} {scale} mean preferred gap',mean(gaps),v,1e-9,source)
 overall=[mean(scale_gaps[(fam,s)][i] for s in scales) for i in range(6)]
 ok(f'{fam} public panel mean preferred gap',mean(overall),.038602 if fam=='DeiT' else .025389,0.0000006,source)
 ok(f'{fam} exact location lower bound',location_bound(overall),.033501 if fam=='DeiT' else .0215,.00005,source)
 count(f'{fam} sign reversal configurations',2**len(overall),64,source)
 ok(f'{fam} exact signflip p one-sided',p_positive(overall),1/64,1e-12,source)

lookup={(x['model_family'],x['scale'],int(x['cohort_index']),x['operator']):float(x['analytic_sigma']) for x in null}
expected_z={('DeiT','ROT180'):(4.35,12),('DeiT','HFLIP'):(.02,4),('DeiT','CLEAN'):(-.83,6),('Swin','ROT180'):(2.08,8),('Swin','HFLIP'):(3.99,17),('Swin','CLEAN'):(-2.69,11)}
for (fam,arm),(m,above) in expected_z.items():
 arr=deit if fam=='DeiT' else swin
 vals=[]
 for x in arr:
  if x.get('operator',x.get('arm'))!=arm:continue
  if fam=='DeiT':z=float(x['q_z_analytic'])
  else:z=float(x['q_plusF'])/lookup[('Swin mixed RoPE',x['scale'],int(x['cohort']),arm)]
  vals.append(z)
 ok(f'{fam} {arm} analytic z mean',mean(vals),m,.0051,'DATA_TABLES/finite_null_analytic_sigma_186_states.csv')
 count(f'{fam} {arm} |z|>1.96 count',sum(abs(x)>1.96 for x in vals),above,'same')
chi=[78*(float(x['empirical_sd_ddof1'])/float(x['analytic_sigma']))**2 for x in null]
ks=kstest(chi,chi2(78).cdf)
count('conditional arrangement-null calibration states',len(chi),186,'DATA_TABLES/finite_null_analytic_sigma_186_states.csv')
ok('null calibration chi2 sample mean',mean(chi),77.616,.0006,'DATA_TABLES/finite_null_analytic_sigma_186_states.csv')
ok('null calibration KS D',ks.statistic,.03619,.000006,'same')
ok('null calibration KS nominal p',ks.pvalue,.9606,.00006,'same')
ok('null calibration SD(sd1/sigma)',stdev(float(x['empirical_sd_ddof1'])/float(x['analytic_sigma']) for x in null),.0750,.00006,'same')
ok('null median sigma/sd1',median(float(x['analytic_sigma'])/float(x['empirical_sd_ddof1']) for x in null),1.009757,.000001,'same')
ok('null median sigma/sd0',median(float(x['analytic_sigma'])/float(x['empirical_sd_ddof0']) for x in null),1.016209,.000001,'same')

# Seed-level q, exact p and inverted L95 from the actual 288 state rows
state=rows('DATA_TABLES/trained_base_288_state_q.csv')
by=defaultdict(dict)
for x in state:by[(x['family'],int(x['checkpoint_index']),int(x['cohort']))][x['operator']]=float(x['q_plusF'])
expected={'DeiT':(.049584,.043088),'Swin':(.042796,.032009)}
for fam in ['DeiT','Swin']:
 vals=[]
 for i in range(1,7):
  qs=[by[(fam,i,j)] for j in range(1,7)]
  pref=mean((a['ROT180']-a['HFLIP'])*(1 if fam=='DeiT' else -1) for a in qs)
  vals.append(pref)
  count(f'{fam} P{i} positive paired cohorts',sum((a['ROT180']-a['HFLIP'])*(1 if fam=='DeiT' else -1)>0 for a in qs),6,'DATA_TABLES/trained_base_288_state_q.csv')
 ok(f'{fam} trained six-seed preferred mean',mean(vals),expected[fam][0],.0000006,'DATA_TABLES/trained_base_288_state_q.csv')
 ok(f'{fam} trained lower 95 bound',location_bound(vals),expected[fam][1],.0000006,'same')
 ok(f'{fam} trained signflip exact p',p_positive(vals),1/64,1e-12,'same')
 if fam=='DeiT':ok('DeiT across-seed standardized contrast',mean(vals)/stdev(vals),3.9,.1,'same')
 else:ok('Swin across-seed standardized contrast',mean(vals)/stdev(vals),3.4,.1,'same')

# S5 state and full-50k Top1 downstream damage
states=rows('DATA_TABLES/S5_AUDITED_144_STATES_20261105.csv')
seed=rows('DATA_TABLES/S5_AUDITED_12_SEED_CONTRASTS_20261105.csv')
pairs=rows('DATA_TABLES/S5_AUDITED_72_COHORT_CONTRASTS_20261105.csv')
count('S5 original arm states',len(states),144,'DATA_TABLES/S5_AUDITED_144_STATES_20261105.csv')
count('S5 state-image evaluations',sum(int(x['n_images']) for x in states),7_200_000,'same')
count('S5 paired damage contrasts positive',sum(float(x['contrast_rot180_minus_hflip_pp'])>0 for x in pairs),72,'DATA_TABLES/S5_AUDITED_72_COHORT_CONTRASTS_20261105.csv')
for fam,(rot,flip,diff,sd) in {'DeiT':(1.7777,.9582,.8195,.6284),'Swin':(.9791,.5515,.4276,.1022)}.items():
 s=[x for x in seed if x['family']==fam]
 for col,target in [('mean_rot180_damage_pp',rot),('mean_hflip_damage_pp',flip),('mean_contrast_pp',diff)]:ok(f'{fam} S5 mean {col}',mean(float(x[col]) for x in s),target,.000051,'DATA_TABLES/S5_AUDITED_12_SEED_CONTRASTS_20261105.csv')
 ok(f'{fam} S5 six-seed SD',stdev(float(x['mean_contrast_pp']) for x in s),sd,.000051,'same')
 if fam=='DeiT':ok('DeiT five-seed excluding P3',mean(float(x['mean_contrast_pp']) for x in s if x['pair']!='P3'),.5639,.00006,'same')

# S7 sample SD and clean accuracy
acc=rows('DATA_TABLES/S7_TRAINED_VS_PUBLIC_12_PAIRS.csv')
for fam,(tar,public,delta,sd) in {'DeiT':(83.7367,83.774,-.0373,.0747),'Swin':(83.8330,83.754,.0790,.0975)}.items():
 a=[float(x['trained_top1_pct']) for x in acc if x['architecture']==fam]
 ok(f'{fam} clean mean Top-1',mean(a),tar,.000051,'DATA_TABLES/S7_TRAINED_VS_PUBLIC_12_PAIRS.csv')
 ok(f'{fam} public Base Top-1',float(next(x['public_top1_pct'] for x in acc if x['architecture']==fam)),public,.000001,'same')
 ok(f'{fam} clean relative difference',mean(a)-public,delta,.000051,'same')
 ok(f'{fam} clean across seed sample SD',stdev(a),sd,.000051,'same')

# A6 fresh, coordinate and stage breakdown
fresh=rows('A6_DIRECTIONAL/A6_84_PAIRED_Q_CONTRASTS.csv')
stage=rows('A6_DIRECTIONAL/A6_SWIN_168_STAGE_PAIR_CONTRASTS.csv')
for fam,(contrast,public,xcon,ycon) in {'DeiT':(.049795,.058610,.029692,.020103),'Swin':(-.042815,-.032594,-.036857,-.005959)}.items():
 a=[r for r in fresh if r['architecture']==fam and r['source']=='TRAINED']
 pub=[r for r in fresh if r['architecture']==fam and r['source']=='PUBLIC']
 for label,xx,tgt in [('trained mean',a,contrast),('public mean',pub,public)]:ok(f'{fam} A6 {label}',mean(float(r['contrast_rot180_minus_hflip']) for r in xx),tgt,.0000006,'A6_DIRECTIONAL/A6_84_PAIRED_Q_CONTRASTS.csv')
 count(f'{fam} A6 preferred signs',sum((float(r['contrast_rot180_minus_hflip'])*(1 if fam=='DeiT' else -1))>0 for r in a),36,'same')
 for label,col,target in [('x','contrast_x_contribution',xcon),('y','contrast_y_contribution',ycon)]:ok(f'{fam} A6 {label} global projection',mean(float(r[col]) for r in a),target,.0000006,'same')
 if fam=='DeiT': ok('DeiT A6 x share pct',100*mean(float(r['contrast_x_contribution']) for r in a)/mean(float(r['contrast_rot180_minus_hflip']) for r in a),59.6,.07,'same')
 else:ok('Swin A6 x share pct',100*mean(float(r['contrast_x_contribution']) for r in a)/mean(float(r['contrast_rot180_minus_hflip']) for r in a),86.1,.07,'same')
for s,(expected_q,energy) in {1:(.002209,.0569),2:(-.001310,.0768),3:(-.011510,.5575),4:(-.032205,.3088)}.items():
 a=[x for x in stage if x['source']=='TRAINED' and int(x['stage'])==s]
 ok(f'Swin A6 stage{s} mean global contrast',mean(float(x['rot180_minus_hflip_stage_contribution']) for x in a),expected_q,.0000006,'A6_DIRECTIONAL/A6_SWIN_168_STAGE_PAIR_CONTRASTS.csv')
 ok(f'Swin A6 stage{s} F-energy fraction',mean(float(x['F_stage_energy_frac']) for x in a),energy,.00006,'same')

# Pairwise location on single public Base DeiT checkpoint
local=rows('DATA_TABLES/deit_locality_six_cohorts.csv')
a=[float(x['global14_rot180_minus_hflip']) for x in local];b=[float(x['tiled7_rot180_minus_hflip']) for x in local]
change=[bb-aa for aa,bb in zip(a,b)]
ok('locality global mean',mean(a),.049266,.0000006,'DATA_TABLES/deit_locality_six_cohorts.csv')
ok('locality tile mean',mean(b),.020706,.0000006,'same')
ok('locality paired change',mean(change),-.028560,.0000006,'same')
ok('locality percent attenuation',100*(mean(a)-mean(b))/mean(a),57.97,.006,'same')
ok('locality upper CI',-location_bound([-v for v in change]),-.015826,.0000006,'same')
count('locality all negative paired changes',sum(x<0 for x in change),6,'same')

# S6 trained initial gradients
fr=rows('DATA_TABLES/trained_checkpoint_first_gradient_288_states.csv')
gr=defaultdict(dict)
for x in fr:
 if x['arm'] in ('ROT180','HFLIP'):
  gr[(x['architecture'],x['pair'],x['cohort'])][x['arm']]=x
for fam,target_sign,target_grad in [('DeiT',.024442,.030879),('Swin',.043817,-.001042)]:
 signv=[];gradv=[]
 for i in range(1,7):
  v=[gr[(fam,'P'+str(i),str(j))] for j in range(1,7)]
  preferred=1 if fam=='DeiT' else -1
  signv.append(mean(preferred*(float(z['ROT180']['a_sign'])-float(z['HFLIP']['a_sign'])) for z in v))
  gradv.append(mean(preferred*(float(z['ROT180']['a_grad'])-float(z['HFLIP']['a_grad'])) for z in v))
 ok(f'{fam} trained first-step sign contrast',mean(signv),target_sign,.0000006,'DATA_TABLES/trained_checkpoint_first_gradient_288_states.csv')
 ok(f'{fam} trained first-step normalized-gradient contrast',mean(gradv),target_grad,.0000006,'same')
 count(f'{fam} trained first-step sign positive seeds',sum(x>0 for x in signv),6,'same')
 if fam=='Swin':count('Swin trained first-step gradient positive seeds',sum(x>0 for x in gradv),4,'same')
swin_hflip=[]
for i in range(1,7):
 swin_hflip.append(mean(float(gr[('Swin',f'P{i}',str(j))]['HFLIP']['a_sign']) for j in range(1,7)))
ok('Swin trained sign-step HFLIP absolute',mean(swin_hflip),.066726,.0000006,'DATA_TABLES/trained_checkpoint_first_gradient_288_states.csv')

payload={'status':'PASS' if all(x['status']=='PASS' for x in checks) else 'FAIL','n_claim_checks':len(checks),'pass':sum(x['status']=='PASS' for x in checks),'failed':[x for x in checks if x['status']!='PASS'],'checks':checks,
'limitations':['Selected numerical claims were audited, not every sentence/figure/table cell in SR3.','Raw 50k ImageNet pixel inference, fresh training, and perturbation generation are out of scope.','Supplement reference F geometry for nine public configs cannot all be regenerated from this package (only 14 Base frequency vectors are included).','Certain historical public first-step and exact sign-flip source rows may require deeper legacy archive extraction.','No complete and independently rendered figure regeneration was performed.']}
out=R/'OUTPUT'/'SR3_NUMERICAL_AUDIT.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
print('SR3_RECOMPUTED_NUMERIC_CHECKS',len(checks),'PASS',payload['pass'],'FAIL',len(payload['failed']))
for v in payload['failed']: print('FAIL',v['name'],v['computed'],v['paper'])
if payload['failed']:sys.exit(1)
print('SR3_SELECTED_NUMERICAL_CLAIMS_PASS')
