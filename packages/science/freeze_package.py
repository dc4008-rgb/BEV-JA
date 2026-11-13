"""Freeze only new scientific source and its fixed diagnostic plans."""
from pathlib import Path
import hashlib,json,os
ROOT=Path(__file__).resolve().parent
files=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p==ROOT/'PACKAGE_MANIFEST.json':continue
    if p.suffix not in ('.py','.md','.json','.txt'):continue
    if p.is_symlink():raise ValueError('new source must not be symlink')
    files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
value={'schema':'ours-periodic-foa-200-package-v1','identity':'ours_periodic_foa_v1','status':'independent_input200_expansion_source','files':files}
with (ROOT/'PACKAGE_MANIFEST.json').open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'manifest_sha256':hashlib.sha256((ROOT/'PACKAGE_MANIFEST.json').read_bytes()).hexdigest(),'members':len(files)}))
