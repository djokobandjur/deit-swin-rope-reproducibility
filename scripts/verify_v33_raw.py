#!/usr/bin/env python3
"""Read-only independent audit of original S5, S6, S7 and A6 archived data.

Does NOT load torch pickles, model weights, ImageNet images or original delta tensors.
Does NOT independently rerun model inference/optimization/gradient capture.
"""
from __future__ import annotations
import csv, hashlib, io, json, math, statistics, sys, zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
Z=ROOT/'ARCHIVES'

def need(cond, why):
    if not cond: raise AssertionError(why)

def sha(b): return hashlib.sha256(b).hexdigest()
def rows(b): return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
def almost(a,b,tol=1e-10):
    need(math.isclose(float(a),float(b),abs_tol=tol,rel_tol=0), f'NOT_CLOSE {a} {b}')

def checked_npz(z,n,doc):
    source=z.read(n)
    need(sha(source)==doc['predictions_sha256'],f'PREDICTION_SHA {n}')
    with np.load(io.BytesIO(source),allow_pickle=False) as arr:
        keys=set(arr.files)
        need({'image_index','y_pred','is_correct'}<=keys and 'y_true' not in keys,f'REDACTED_NPZ_SCHEMA {n}')
        idx=arr['image_index']; pred=arr['y_pred']; valid=arr['is_correct']
        need(idx.shape==pred.shape==valid.shape==(50000,),f'NPZ_SHAPE {n}')
        need(np.array_equal(idx,np.arange(50000)),f'INDEX {n}')
        need(pred.min()>=0 and pred.max()<1000,f'CLASS_RANGE {n}')
        need(set(np.unique(valid).tolist())<= {False,True,0,1},f'CORRECTNESS_RANGE {n}')
        label_hash=doc.get('ground_truth_label_vector_sha256')
        need(isinstance(label_hash,str) and len(label_hash)==64,f'ORIGINAL_LABEL_COMMITMENT {n}')
        correct=int(np.count_nonzero(valid))
        return label_hash,correct

def original_archive_checks():
    with zipfile.ZipFile(Z/'a6/A6_ORIGINAL_168_RAW.zip') as z:
        names=z.namelist()
        need(z.testzip() is None, 'A6_COMPLETE_CRC')
        need(sum('/raw_cases/' in n and n.endswith('/TECHNICAL_SEAL.json') for n in names)==84,'A6_84_CASEPAIRS')
        need(sum('/raw_cases/' in n and n.endswith('_delta.pt') for n in names)==168,'A6_168_DELTA_ARTIFACTS')
        print('A6 archival structure: 84 case pairs / 168 original raw delta tensors; CRC PASS')

def main():
    original_archive_checks()
    with zipfile.ZipFile(Z/'numerical/V33_NUMERICAL_TABLES_ONLY.zip') as manuscript,\
         zipfile.ZipFile(Z/'s5/S5_FULLVAL_144_STATES.zip') as s5,\
         zipfile.ZipFile(Z/'s6/S6_288_GRADIENT_METRICS.zip') as s6,\
         zipfile.ZipFile(Z/'s7/S7_TRAINED_CLEAN_FULLVAL.zip') as s7,\
         zipfile.ZipFile(Z/'s7/S7_PUBLIC_CLEAN_FULLVAL.zip') as s7public:
        prefix='NEUROCOMPUTING_V3_3_RC1_20261106/data_tables/'
        published_states=rows(manuscript.read(prefix+'S5_AUDITED_144_STATES_20261105.csv'))
        published_seeds=rows(manuscript.read(prefix+'S5_AUDITED_12_SEED_CONTRASTS_20261105.csv'))
        published_cohorts=rows(manuscript.read(prefix+'S5_AUDITED_72_COHORT_CONTRASTS_20261105.csv'))
        published_s7=rows(manuscript.read(prefix+'S7_TRAINED_VS_PUBLIC_12_PAIRS.csv'))
        published_s6=rows(manuscript.read(prefix+'trained_checkpoint_first_gradient_288_states.csv'))
        published_s6_seeds=rows(manuscript.read(prefix+'trained_checkpoint_first_gradient_12_seed_contrasts.csv'))
        need((len(published_states),len(published_cohorts),len(published_seeds),len(published_s7),len(published_s6),len(published_s6_seeds))==(144,72,12,12,288,12),'PUBLISHED_ROW_COUNTS')

        # S7 clean NPZ from 12 trained checkpoint instances and 2 public checkpoints.
        clean={}; reference={}; labels=None
        for arch in ['DeiT','Swin']:
            for i in range(1,7):
                name=f'{arch}_P{i}'
                meta=json.loads(s7.read(name+'.json'))
                y,count=checked_npz(s7,name+'.npz',meta)
                need(int(meta['correct'])==count and int(meta['n_images'])==50000,'S7_TRAINED_COUNTS')
                almost(meta['top1_pct'],100*count/50000)
                if labels is None: labels=y
                need(y==labels,'S7_TRAINED_LABEL_STREAM')
                clean[(arch,f'P{i}')]=(meta,count)
            public_name=f'S7_PUBLIC_REFERENCE_v1_20261105/full/PUBLIC_{arch}_Base'
            doc=json.loads(s7public.read(public_name+'.json'))
            y,count=checked_npz(s7public,public_name+'.npz',doc)
            need(int(doc['correct'])==count,'S7_PUBLIC_COUNTS')
            need(y==labels,'S7_PUBLIC_LABEL_STREAM')
            almost(doc['top1_pct'],100*count/50000)
            reference[arch]=(doc,count)
        print('S7 public-safe: 12 trained + 2 public, all 50k, prediction/correctness arrays and shared original label commitment PASS')

        # S5 archive: every NPZ exact-byte verified against corresponding JSON and fullval clean output.
        states={}
        published={r['state']:r for r in published_states}
        need(len(published)==144,'S5_PUBLISHED_DUPLICATE_STATES')
        authority={}
        for r in rows((ROOT/'CODE/S5_FROZEN_144_DELTA_AUTHORITY.csv').read_bytes()):
            key=(r['architecture'],r['pair'],int(r['cohort']),r['arm'])
            need(key not in authority,'S5_AUTH_DUPLICATE')
            authority[key]=r
        need(len(authority)==144,'S5_AUTH_COUNT')
        for arch in ('DeiT','Swin'):
            for i in range(1,7):
                pair=f'P{i}'
                for j in range(1,7):
                    for arm in ('ROT180','HFLIP'):
                        key=(arch,pair,j,arm)
                        name=f'{arch}_{pair}_cohort{j:02d}_{arm}'
                        doc=json.loads(s5.read(name+'.json'))
                        y,count=checked_npz(s5,name+'.npz',doc)
                        need(y==labels,f'S5_LABEL_STREAM {name}')
                        clean_doc,clean_count=clean[(arch,pair)]
                        need(int(doc['perturbed_correct'])==count,f'S5_PERTURBED_COUNT {name}')
                        need(int(doc['clean_correct'])==clean_count,f'S5_CLEAN_COUNT {name}')
                        need(doc['checkpoint_sha256']==clean_doc['checkpoint_sha256'],f'S5_CHECKPOINT_HASH {name}')
                        need(doc['source_clean_prediction_sha256']==clean_doc['predictions_sha256'],f'S5_SOURCE_PRED_HASH {name}')
                        need(doc['fullval_stream_sha256']==clean_doc['cache_semantic_stream_sha256'],f'S5_CACHE_STREAM {name}')
                        almost(doc['damage_pp'],100*(clean_count-count)/50000)
                        need(doc['delta_sha256']==authority[key]['delta_sha256'],f'S5_DELTA_AUTHORITY {name}')
                        need(doc['F_vector_sha256']==authority[key]['F_vector_sha256'],f'S5_F_AUTHORITY {name}')
                        need(doc['checkpoint_sha256']==authority[key]['checkpoint_sha256'],f'S5_CP_AUTHORITY {name}')
                        almost(doc['verified_q_plusF'],authority[key]['q_plusF'],tol=1e-12)
                        pub=published[name]
                        need(pub['delta_sha256']==doc['delta_sha256'] and pub['predictions_sha256']==doc['original_predictions_sha256'],f'S5_PUBLISHED_HASH {name}')
                        almost(pub['verified_q_plusF'],doc['verified_q_plusF'],tol=1e-12)
                        almost(pub['damage_pp'],doc['damage_pp'])
                        states[key]={'q':float(doc['verified_q_plusF']),'damage':float(doc['damage_pp']),'checkpoint_sha256':doc['checkpoint_sha256'],'F_vector_sha256':doc['F_vector_sha256'],'delta_sha256':doc['delta_sha256']}
        need(len(states)==144 and len(s5.namelist())==288,'S5_NPZ_JSON_COUNT')
        print('S5 public-safe: 144 JSON+NPZ state pairs, 7.2M saved correctness outcomes; S7, original-sha provenance and authority agreement PASS (no y_true redistribution)')

        # Paired contrasts per frozen cohort and seed from source predictions.
        pubcoh={(x['family'],x['pair'],int(x['cohort'])):x for x in published_cohorts}
        pubseed={(x['family'],x['pair']):x for x in published_seeds}
        positives=0
        for arch in ['DeiT','Swin']:
            means=[]
            for i in range(1,7):
                pair=f'P{i}'; ds=[]
                for j in range(1,7):
                    diff=states[(arch,pair,j,'ROT180')]['damage']-states[(arch,pair,j,'HFLIP')]['damage']
                    almost(pubcoh[(arch,pair,j)]['contrast_rot180_minus_hflip_pp'],diff)
                    ds.append(diff);positives+=int(diff>0)
                almost(pubseed[(arch,pair)]['mean_contrast_pp'],statistics.mean(ds))
                means.append(statistics.mean(ds))
            print(f'S5 {arch}: mean ROT180−HFLIP damage {statistics.mean(means):+.9f} pp, seed SD {statistics.stdev(means):.9f}')
        need(positives==72,'S5_COHORT_SIGN')

        # S6: immutable original data vs published table; raw 144 physical q and SHA match S5.
        s6_rows=rows(s6.read('FIRST_GRADIENT_288_STATE_METRICS.csv'))
        s6_seeds=rows(s6.read('FIRST_GRADIENT_12_SEED_CONTRASTS.csv'))
        need(len(s6_rows)==288 and len(s6_seeds)==12,'S6_ROW_COUNTS')
        need(s6_rows==published_s6,'S6_PUBLISHED_METRIC_TABLE_DRIFT')
        need(s6_seeds==published_s6_seeds,'S6_PUBLISHED_SEED_TABLE_DRIFT')
        subset={}
        for r in s6_rows:
            key=(r['architecture'],r['pair'],int(r['cohort']),r['arm'])
            need(key not in subset,'S6_DUPLICATE_KEY')
            subset[key]=r
            need(r['G2_pass']=='True' and r['G3_pass']=='True','S6_ORIGINAL_GATE')
            if key in states:
                state=states[key]
                for f in ('checkpoint_sha256','F_vector_sha256','delta_sha256'):
                    need(r[f]==state[f],f'S6_{f}_{key}')
                almost(r['q'],state['q'],1e-12)
        need(len(subset)==288 and all(key in subset for key in states),'S6_MATCHED_144')
        for x in s6_seeds:
            arch,pair=x['architecture'],x['pair']
            first,second=(('ROT180','HFLIP') if arch=='DeiT' else ('HFLIP','ROT180'))
            for measure,prefix_ in [('q','q'),('a_grad','a_grad'),('a_sign','a_sign')]:
                vals=[float(subset[(arch,pair,j,first)][measure])-float(subset[(arch,pair,j,second)][measure]) for j in range(1,7)]
                almost(statistics.mean(vals),x[prefix_+'_seed_mean'],tol=1e-12)
                need(sum(t>0 for t in vals)==int(x[prefix_+'_positive_cohorts']),'S6_POSITIVE_COUNTS')
        print('S6 original: 288 rows, 12 seed means, 144 S5-matched q/checkpoint/F/delta hashes PASS')

        # Published 12 trained vs public clean validation comparisons computed independently from the predictions.
        for r in published_s7:
            arch,pair=r['architecture'],r['trained_replicate']
            cp,correct=clean[(arch,pair)]
            publicdoc,ref_correct=reference[arch]
            almost(r['trained_top1_pct'],100*correct/50000)
            almost(r['public_top1_pct'],100*ref_correct/50000)
            almost(r['trained_minus_public_pp'],100*(correct-ref_correct)/50000)
            need(r['trained_checkpoint_sha256']==cp['checkpoint_sha256'],'S7_TRAINED_CHECKPOINT_SHA')
            need(r['public_checkpoint_sha256']==publicdoc['checkpoint_sha256'],'S7_PUBLIC_CHECKPOINT_SHA')
        print('Cross-archive v3.3 S7 12 trained-versus-public comparisons PASS')
    print('ADSV3_V33_RAW_ARTIFACT_REPRO_PASS')

if __name__=='__main__':
    main()
