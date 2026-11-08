#!/usr/bin/env python3
"""CPU-only, artifact-backed A6 RoPE-F directional / XY / Swin-stage analysis.

Inputs:
  F-export ZIP (14 checkpoint-sourced F vectors, frozen STEP9 byte serialization),
  A6 original raw-artifact ZIP (168 physically saved torch deltas and SHA seals).
No model forward, GPU, retraining, or new perturbation generation.

Loading of research-produced .pt uses torch.load(..., weights_only=True).
Analyses are retrospective/exploratory relative to original A6 lock.
"""
import argparse, csv, hashlib, io, json, re, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

NEED_JOBS = [f'DEIT_P{i}_FINETUNE' for i in range(1,7)] + [f'SWIN_P{i}' for i in range(1,7)] + ['PUBLIC_DEIT_BASE','PUBLIC_SWIN_BASE']
ARMS = ('HFLIP','ROT180')

def require(ok, msg):
    if not ok: raise AssertionError(msg)

def digest(b): return hashlib.sha256(b).hexdigest()

def arr_sha(a): return digest(np.ascontiguousarray(a, dtype='<f8').tobytes(order='C'))

def record(job):
    if job.startswith('DEIT_P'):
        return 'DeiT_P' + job[len('DEIT_P'):].split('_')[0]
    if job.startswith('SWIN_P'):
        return 'Swin_P' + job[len('SWIN_P'):]
    return {'PUBLIC_DEIT_BASE':'DeiT_PUBLIC','PUBLIC_SWIN_BASE':'Swin_PUBLIC'}[job]

def compute(a6_zip, f_zip, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(f_zip) as zf:
        require(zf.testzip() is None, 'BAD_F_ZIP')
        fspec=json.loads(zf.read('F_VECTOR_MANIFEST.json'))
        rawf=zf.read('ADSV3_14_F_VECTORS.npz')
        fv=np.load(io.BytesIO(rawf), allow_pickle=False)
        require(fspec['trained_F_hashes_checked']==12 and fspec['public_checkpoint_hashes_checked']==2, 'F_AUTHORITY_CHECKS')
        fmodels={m['model_id']:m for m in fspec['models']}
        require(len(fmodels)==14 and len(fv.files)==42, 'F_COMPLETENESS')
        for mid,m in fmodels.items():
            f=fv[mid+'_F']; fx=fv[mid+'_Fx']; fy=fv[mid+'_Fy']; parts=m['parts']
            require(arr_sha(f)==m['F_sha256_float64_le'],f'F_SHA_{mid}')
            require(f.shape==(m['F_numel'],) and np.isfinite(f).all(), f'F_SHAPE_{mid}')
            pos=0; ax=[]; ay=[]
            for p in parts:
                n=p['float64_numel']; sh=p['shape']
                require(n==np.prod(sh) and sh[0]==2, f'F_PART_SHAPE_{mid}')
                v=f[pos:pos+n].reshape(sh)
                require(arr_sha(v)==p['sha256_float64_le'],f'F_PART_SHA_{mid}')
                ax.append(v[0].ravel()); ay.append(v[1].ravel()); pos+=n
            require(pos==len(f) and np.array_equal(np.concatenate(ax),fx) and np.array_equal(np.concatenate(ay),fy),f'F_XY_{mid}')
        f_authority_sha=digest(zf.read('checkpoint_authority_snapshot.csv'))
        require(f_authority_sha==fspec['source_authority_file_sha256'],'AUTHORITY_SNAPSHOT_SHA')

        with zipfile.ZipFile(a6_zip) as za:
            require(za.testzip() is None, 'BAD_A6_ZIP')
            paths=za.namelist()
            root='ADSV3_A6_RAW_ARTIFACT_REPRO_COMPLETE_20261025/'
            cases=list(csv.DictReader(io.StringIO(za.read(root+'locked/A6_LOCKED_168_CASES.csv').decode())))
            require(len(cases)==168 and len({(r['job_id'],r['cohort'],r['arm']) for r in cases})==168,'A6_AUTHORITY_168')
            require(set(r['job_id'] for r in cases)=={('PUBLIC_NAVER_'+j[len('PUBLIC_'):] if j.startswith('PUBLIC_') else j) for j in NEED_JOBS},'A6_14_JOBS')
            identity=list(csv.DictReader(io.StringIO(za.read(root+'provenance/A6_FRESH_1536_IDENTITY_MAP.csv').decode())))
            historical=list(csv.DictReader(io.StringIO(za.read(root+'locked/HISTORICAL_MCP1_COHORT_LOCK.csv').decode())))
            fresh={int(r['val_index0']) for r in identity}; old={int(r['val_index0']) for r in historical}
            require(len(identity)==len(fresh)==len(historical)==len(old)==1536,'A6_IDENTITY_UNIQUE')
            require(not (fresh & old),'A6_FRESH_HISTORICAL_OVERLAP')
            require(all('label' not in r and 'wnid' not in r and 'relative_path' not in r for r in identity),'A6_REDACTED_IDENTITY_SCHEMA'); classes=None # original source records 777 classes, not independently recalculable without ImageNet annotations
            rows=[];stage_rows=[]; q_xcheck_err=0.0; npz_sha_checks=0; all_ck=0
            for job in NEED_JOBS:
                mid=record(job); m=fmodels[mid]; f=fv[mid+'_F']; f_norm=np.linalg.norm(f)
                parts=m['parts']; ps=[p['key'] for p in parts]
                require(ps==sorted(ps),'F_KEY_ORDER')
                for cohort in range(1,7):
                    folder=f'{root}raw_cases/{job}/cohort_{cohort:02d}/'
                    seal=json.loads(za.read(folder+'TECHNICAL_SEAL.json'))
                    require(seal['job_id']==job and int(seal['cohort'])==cohort, 'SEAL_JOB_COHORT')
                    require(seal['checkpoint_sha256']==m['checkpoint_sha256'],f'CHECKPOINT_SHA_SEAL_{job}')
                    require(seal['status']=='PASS_E2E_TECHNICAL_SCIENCE_PAYLOAD_UNOPENED','A6_TECH_SEAL_STATUS')
                    all_ck+=1
                    for name,expected in seal['payload_files_sha256'].items():
                        b=za.read(folder+name)
                        require(digest(b)==expected,f'A6_PAYLOAD_SHA_{job}_{cohort}_{name}')
                        npz_sha_checks+=1
                    for arm in ARMS:
                        name=f'{arm}_delta.pt';b=za.read(folder+name)
                        t=torch.load(io.BytesIO(b), map_location='cpu', weights_only=True)
                        require(t['arm']==arm and int(t['cohort'])==cohort, 'DELTA_ID')
                        require(int(t['steps'])==20 and abs(float(t['epsilon'])-0.2)<1e-7 and abs(float(t['beta'])-0.1)<1e-7 and abs(float(t['alpha'])-0.02)<1e-7,'DELTA_PARAMS')
                        if mid.startswith('DeiT'):
                            ts={'freqs':t['delta']}
                        else:
                            ts=t['delta_by_key']
                            require(t['surface_keys']==sorted(t['surface_keys']),'A6_DELTA_SURFACE_KEYS_ORDER')
                        require(list(ts.keys())==ps, f'DELTA_VS_F_KEYS_{mid}')
                        dlist=[];pd=[]
                        for p in parts:
                            x=ts[p['key']]
                            require(isinstance(x,torch.Tensor) and tuple(x.shape)==tuple(p['shape']) and x.dtype==torch.float32, 'DELTA_TENSOR_SHAPE_DTYPE')
                            require(torch.isfinite(x).all().item() and float(x.abs().max())<=0.200001,'DELTA_FINITE_BOUND')
                            a=x.numpy().astype(np.float64,copy=False)
                            dlist.append(a.ravel());pd.append(a)
                        d=np.concatenate(dlist);require(len(d)==len(f),'D_F_LENGTH')
                        norm=np.linalg.norm(d); denom=norm*f_norm
                        require(denom>0,'ZERO_F_OR_D')
                        q=float(d@f/denom)
                        dotx=0.;doty=0.; d2x=0.;d2y=0.;f2x=0.;f2y=0.
                        pos=0; stg=defaultdict(float); stg_f2=defaultdict(float)
                        for p,a in zip(parts,pd):
                            sh=p['shape']; n=p['float64_numel']; fpart=f[pos:pos+n].reshape(sh)
                            dx=a[0];dy=a[1];fx=fpart[0];fy=fpart[1]
                            dotx+=float(np.sum(dx*fx));doty+=float(np.sum(dy*fy))
                            d2x+=float(np.sum(dx**2));d2y+=float(np.sum(dy**2))
                            f2x+=float(np.sum(fx**2));f2y+=float(np.sum(fy**2))
                            if mid.startswith('Swin'):
                                stage=int(re.match(r'layers\.(\d+)\.',p['key']).group(1))+1
                                stg[stage]+=float(np.sum(a*fpart));stg_f2[stage]+=float(np.sum(fpart**2))
                            pos+=n
                        require(pos==len(f),'PART_OFFSET')
                        q_x=dotx/denom; q_y=doty/denom
                        q_xcheck_err=max(q_xcheck_err,abs(q-(q_x+q_y)))
                        require(abs(q-(q_x+q_y))<1e-13,f'Q_SUM_XY_{mid}_{cohort}_{arm}')
                        row={'model':mid,'job_id':job,'architecture':m['architecture'],'source':('PUBLIC' if m['public'] else 'TRAINED'),
                             'cohort':cohort,'arm':arm,'n_images':256,'delta_sha256':digest(b),
                             'checkpoint_sha256':m['checkpoint_sha256'],'F_sha256':m['F_sha256_float64_le'],
                             'F_numel':len(f),'norm_delta':norm,'norm_F':f_norm,'q':q,
                             'q_x_contribution':q_x,'q_y_contribution':q_y,
                             'q_x_axis_cosine':dotx/np.sqrt(d2x*f2x) if d2x*f2x>0 else np.nan,
                             'q_y_axis_cosine':doty/np.sqrt(d2y*f2y) if d2y*f2y>0 else np.nan,
                             'F_x_energy_fraction':f2x/(f2x+f2y),
                             'delta_x_energy_fraction':d2x/(d2x+d2y),
                             'F_projection_delta_energy_fraction':q*q}
                        rows.append(row)
                        for s in sorted(stg):
                            stage_rows.append({'model':mid,'source':row['source'],'architecture':'Swin',
                                               'cohort':cohort,'arm':arm,'stage':s,
                                               'q_stage_contribution':stg[s]/denom,'F_stage_energy_frac':stg_f2[s]/(f_norm*f_norm)})
            require(len(rows)==168 and len(stage_rows)==84*4,'ALL_A6_CASES')
            rby={(r['model'],r['cohort'],r['arm']):r for r in rows}
            contrasts=[];sc=[]
            for mid in sorted(fmodels):
                for cohort in range(1,7):
                    a=rby[mid,cohort,'ROT180'];b=rby[mid,cohort,'HFLIP'];
                    contrasts.append({'model':mid,'architecture':a['architecture'],'source':a['source'],'cohort':cohort,
                                      'q_rot180':a['q'],'q_hflip':b['q'],
                                      'contrast_rot180_minus_hflip':a['q']-b['q'],
                                      'contrast_x_contribution':a['q_x_contribution']-b['q_x_contribution'],
                                      'contrast_y_contribution':a['q_y_contribution']-b['q_y_contribution'],
                                      'rot180_q_x_axis_cosine':a['q_x_axis_cosine'],
                                      'hflip_q_x_axis_cosine':b['q_x_axis_cosine'],
                                      'rot180_q_y_axis_cosine':a['q_y_axis_cosine'],
                                      'hflip_q_y_axis_cosine':b['q_y_axis_cosine']})
                rr=[r for r in contrasts if r['model']==mid]
                sc.append({'model':mid,'architecture':rr[0]['architecture'],'source':rr[0]['source'],
                           'mean_rot180':float(np.mean([r['q_rot180'] for r in rr])),
                           'mean_hflip':float(np.mean([r['q_hflip'] for r in rr])),
                           'mean_rot180_minus_hflip':float(np.mean([r['contrast_rot180_minus_hflip'] for r in rr])),
                           'mean_x_contribution':float(np.mean([r['contrast_x_contribution'] for r in rr])),
                           'mean_y_contribution':float(np.mean([r['contrast_y_contribution'] for r in rr])),
                           'positive_cohorts':sum(r['contrast_rot180_minus_hflip']>0 for r in rr),
                           'n_cohorts':len(rr)})

            stagecontrasts=[]
            for mid in sorted(fmodels):
                if not mid.startswith('Swin'):continue
                for cohort in range(1,7):
                    for stage in range(1,5):
                        a=next(r for r in stage_rows if r['model']==mid and r['cohort']==cohort and r['arm']=='ROT180' and r['stage']==stage)
                        b=next(r for r in stage_rows if r['model']==mid and r['cohort']==cohort and r['arm']=='HFLIP' and r['stage']==stage)
                        stagecontrasts.append({'model':mid,'source':a['source'],'cohort':cohort,'stage':stage,
                                               'rot180_minus_hflip_stage_contribution':a['q_stage_contribution']-b['q_stage_contribution'],
                                               'F_stage_energy_frac':a['F_stage_energy_frac']})
                for cohort in range(1,7):
                    part=sum(r['rot180_minus_hflip_stage_contribution'] for r in stagecontrasts if r['model']==mid and r['cohort']==cohort)
                    full=next(r['contrast_rot180_minus_hflip'] for r in contrasts if r['model']==mid and r['cohort']==cohort)
                    require(abs(part-full)<1e-13,'STAGE_CONTRIBUTION_SUM')

            def save_csv(fname,data):
                with (outdir/fname).open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f, fieldnames=list(data[0]));w.writeheader();w.writerows(data)
            save_csv('A6_168_STATE_Q_XY.csv',rows)
            save_csv('A6_84_PAIRED_Q_CONTRASTS.csv',contrasts)
            save_csv('A6_14_MODEL_MEAN_CONTRASTS.csv',sc)
            save_csv('A6_SWIN_336_STAGE_STATE_CONTRIBUTIONS.csv',stage_rows)
            save_csv('A6_SWIN_168_STAGE_PAIR_CONTRASTS.csv',stagecontrasts)
            trained={}
            for arch in ['DeiT','Swin']:
                r=[x for x in sc if x['architecture']==arch and x['source']=='TRAINED']
                vs=np.asarray([x['mean_rot180_minus_hflip'] for x in r])
                # Sign-flip enumeration on independent seed-level contrasts; no post-hoc confirmatory designation.
                import itertools
                means=np.array([np.mean(vs*np.array(s)) for s in itertools.product((-1,1), repeat=6)])
                mean=float(np.mean(vs));direction=1 if mean>=0 else -1
                pdir=float(np.mean(means*direction >= mean*direction-1e-15))
                trained[arch]={'mean_q_ROT180_minus_HFLIP':mean,'sd_seed':float(np.std(vs,ddof=1)),
                    'positive_seeds':int(np.sum(vs>0)),'n_seeds':6,'positive_cohorts':sum(x['contrast_rot180_minus_hflip']>0 for x in contrasts if x['architecture']==arch and x['source']=='TRAINED'),
                    'n_cohort_pairs':36,'orientation':'ROT180>HFLIP' if mean>0 else 'HFLIP>ROT180',
                    'exploratory_one_sided_sign_flip_p':pdir,
                    'mean_x_contribution':float(np.mean([x['mean_x_contribution'] for x in r])),
                    'mean_y_contribution':float(np.mean([x['mean_y_contribution'] for x in r])),
                    'model_seed_contrasts':{x['model']:x['mean_rot180_minus_hflip'] for x in r}}
            public={}
            for r in sc:
                if r['source']!='PUBLIC': continue
                public[r['architecture']]={k:v for k,v in r.items() if k not in ('model','source','architecture')}
            stage_summary=[]
            for mid in sorted(fmodels):
                if not mid.startswith('Swin'):continue
                for stage in range(1,5):
                    subset=[r for r in stagecontrasts if r['model']==mid and r['stage']==stage]
                    stage_summary.append({'model':mid,'source':subset[0]['source'],'stage':stage,
                        'mean_rot180_minus_hflip_stage':float(np.mean([r['rot180_minus_hflip_stage_contribution'] for r in subset])),
                        'mean_F_stage_energy':float(np.mean([r['F_stage_energy_frac'] for r in subset])),
                        'positive_cohorts':sum(r['rot180_minus_hflip_stage_contribution']>0 for r in subset),'n_cohorts':6})
            save_csv('A6_SWIN_STAGE_MODEL_MEANS.csv',stage_summary)
            results={'status':'A6_F_DIRECTIONAL_RETROSPECTIVE_REANALYSIS_PASS',
                     'scientific_status':'POST_HOC_RETROSPECTIVE_NEW_READOUT_NONCONFIRMATORY',
                     'a6_original_cohorts':6,'a6_fresh_images':1536,'a6_fresh_class_count_recomputed':classes,
                     'trained_model_n':12,'public_model_n':2,'state_n':168,'pair_n':84,
                     'source_payload_files_sha_verified':npz_sha_checks,
                     'fresh_image_overlap_with_historical':len(fresh & old),
                     'source_seals_checked':all_ck,'F_vectors_checked':14,
                     'max_xy_sum_error':q_xcheck_err,
                     'trained':trained,'public':public,
                     'F_source_export_sha256':digest(Path(f_zip).read_bytes()),
                     'A6_source_raw_zip_sha256':digest(Path(a6_zip).read_bytes())}
            (outdir/'A6_DIRECTIONAL_RESULTS.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
            print(json.dumps(results,indent=2))
            print('DONE',len(rows),'states',len(contrasts),'contrasts',len(stage_rows),'stage-states')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fzip',type=Path,required=True)
    p.add_argument('--a6zip',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); compute(args.a6zip,args.fzip,args.out)
