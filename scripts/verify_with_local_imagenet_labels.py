#!/usr/bin/env python3
"""Optional full per-image y_pred/accuracy validation with *user-supplied* lawful ImageNet labels.

This tool does not download, create, include, cache or redistribute ImageNet labels.
Supply a local 50,000-entry integer NumPy array in the exact source-native
sorted ImageFolder validation order used in the original experiment.
"""
import argparse, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
par=argparse.ArgumentParser()
par.add_argument('--labels-npy',required=True,help='Your LOCAL authorized 50k ImageNet validation labels .npy (never upload to Zenodo)')
args=par.parse_args()
labels=np.load(args.labels_npy,allow_pickle=False)
assert labels.shape==(50000,) and np.issubdtype(labels.dtype,np.integer), 'EXPECTED_50000_INT_LABELS'
labels=np.asarray(labels,dtype='<i8')
assert labels.min()>=0 and labels.max()<1000, 'IMAGENET_CLASS_RANGE'
label_sha=hashlib.sha256(labels.tobytes()).hexdigest()
passed=0
for name in ('ARCHIVES/s7/S7_TRAINED_CLEAN_FULLVAL.zip',
             'ARCHIVES/s7/S7_PUBLIC_CLEAN_FULLVAL.zip',
             'ARCHIVES/s5/S5_FULLVAL_144_STATES.zip'):
    with zipfile.ZipFile(root/name) as z:
        for member in z.namelist():
            if not member.endswith('.npz') or '/pilot/' in member:continue
            doc=json.loads(z.read(member[:-4]+'.json'))
            assert doc['ground_truth_label_vector_sha256']==label_sha,('LABEL_STREAM_SHA_MISMATCH',member)
            with np.load(io.BytesIO(z.read(member)),allow_pickle=False) as a:
                assert 'y_true' not in a.files and {'y_pred','is_correct','image_index'}<=set(a.files)
                assert np.array_equal(a['image_index'],np.arange(50000)),member
                assert np.array_equal(a['is_correct'],a['y_pred']==labels),('PREDICTION_CORRECTNESS_MISMATCH',member)
            passed+=1
assert passed==158,('EXPECTED_144_S5_PLUS_14_S7',passed)
print('AUTHORIZED_IMAGENET_LABEL_VALIDATION_PASS',passed,'/ 158 complete model-state streams',flush=True)
