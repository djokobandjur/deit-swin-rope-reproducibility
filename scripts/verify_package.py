#!/usr/bin/env python3
from pathlib import Path
import hashlib,zipfile
root=Path(__file__).resolve().parents[1]
entries=[x.strip().split('  ',1) for x in (root/'SHA256SUMS.txt').read_text().splitlines() if x.strip()]
for digest,name in entries:
 p=root/name
 assert p.is_file(),name
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 assert h==digest,(name,digest,h)
 if p.suffix=='.zip':
  with zipfile.ZipFile(p) as z: assert z.testzip() is None,name
print(f'INTEGRITY_PASS {len(entries)}/{len(entries)} files and all source ZIP CRCs')
