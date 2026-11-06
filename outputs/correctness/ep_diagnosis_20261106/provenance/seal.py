"""Seal the exact source/checkpoint used by all four final captures."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

root = Path.cwd()
names = ['final_bf16_tp', 'final_bf16_ep', 'final_precise_tp', 'final_precise_ep']
manifests = [json.loads((root / n / 'manifest.json').read_text()) for n in names]
for m in manifests[1:]:
    assert m['source_sha256'] == manifests[0]['source_sha256']
    assert m['checkpoint_sha256'] == manifests[0]['checkpoint_sha256']
for name, expected in manifests[0]['source_sha256'].items():
    assert hashlib.sha256((root / 'v9' / name).read_bytes()).hexdigest() == expected, name
with tarfile.open('capture_source.tar.gz', 'w:gz') as tar:
    for name in manifests[0]['source_sha256']:
        tar.add(root / 'v9' / name, arcname='v9/' + name)
with tarfile.open('synthetic_checkpoint.tar.gz', 'w:gz') as tar:
    tar.add('tiny', arcname='tiny')
environment = dict(capture_cases=names, all_capture_source_hashes_match=True,
                   all_capture_checkpoint_hashes_match=True,
                   driver=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,driver_version',
                                                   '--format=csv'], text=True))
Path('verified_environment.json').write_text(json.dumps(environment, indent=2))
print(json.dumps(environment, indent=2))
