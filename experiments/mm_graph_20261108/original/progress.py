from pathlib import Path
import json,time
ROOT=Path('/data/ifly_jyhe23/vllm-learn-mm-benchmark-v2-20261108')
r=ROOT/'outputs/completion_suite'
job=json.loads((r/'resume_job.json').read_text())
print('SUITE',job['status'],'elapsed_s',round(time.time()-job['started']))
print((r/'resume.log').read_text()[-600:])
for f in r.glob('*/case.json'):
 d=json.loads(f.read_text())
 if d['status']=='running':print('STAGE',f.parent.name,'elapsed_s',round(time.time()-d.get('started',d.get('start',time.time()))),'pid',d['pid'])
paths=sorted((ROOT/'outputs/load_matrix').glob('*/case.json'))
passed=[]
for f in paths:
 d=json.loads(f.read_text())
 if d['status']=='passed':passed.append(f.parent.name)
 elif d['status']=='running':
  print('CASE',f.parent.name,'elapsed_s',round(time.time()-d['start']),'pid',d['pid'])
  p=f.parent/'results.json'
  if p.exists():
   for rep in json.loads(p.read_text()):print('ROUND',rep['repeat'],'completed',rep['completed'],'errors',rep['errors'],'tok_s',round(rep['output_tok_s'],3))
 else:print('CASE_FAILED',f.parent.name,(f.parent/'run.log').read_text()[-2500:])
print('PASSED_CASES',len(passed),passed)
if job['status']=='failed':
 for f in r.glob('*/case.json'):
  if json.loads(f.read_text())['status']=='failed':print('FAILED',f.parent.name,(f.parent/'run.log').read_text()[-3500:])
