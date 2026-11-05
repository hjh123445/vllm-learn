import argparse,json,pathlib,sys,torch
from analyze_numerical_diagnostics import compare
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--reference-dir');a=p.parse_args();root=pathlib.Path(a.root)
refdir=pathlib.Path(a.reference_dir) if a.reference_dir else root/'hf_cached_teacher'
ref=json.loads((refdir/'text_reference.json').read_text());refs={}
for r in ref:
 z=torch.load(pathlib.Path(r['raw_logits_file']),weights_only=True)
 for i,v in enumerate(z):refs[(r['workload'],r['index'],i)]=v
result={};captured={}
for folder in root.glob('cached_teacher_*'):
 if not (folder/'requests.json').exists():continue
 samples=[]
 for f in sorted((folder/'captures').glob('*.pt')):
  b=torch.load(f,weights_only=True)
  samples.extend(zip(b['seq_ids'],b['logits']))
 requests=json.loads((folder/'requests.json').read_text());assert len(samples)==len(requests),(folder,len(samples),len(requests))
 rows=[];captured[folder.name]={}
 by_seq={}
 for sid,logits in samples:by_seq.setdefault(sid,[]).append(logits)
 for req in requests:
  logits=by_seq[req['seq_id']][req['prefix_tokens']]
  key=(req['workload'],req['index'],req['prefix_tokens']);captured[folder.name][key]=logits
  rows.append(dict(workload=key[0],index=key[1],prefix_tokens=key[2],**compare(logits,refs[key])))
 result[folder.name]=dict(cases=len(rows),matches=sum(r['top1_matches'] for r in rows),worst_abs=max(r['max_abs_error'] for r in rows),records=rows)
cross={}
for name,values in captured.items():
 if name=='cached_teacher_ep1':continue
 if 'cached_teacher_ep1' not in captured:continue
 rows=[dict(key=list(k),**compare(v,captured['cached_teacher_ep1'][k])) for k,v in values.items()]
 cross[name]=dict(cases=len(rows),matches=sum(r['top1_matches'] for r in rows),records=rows)
(root/'cached_teacher_diagnostics.json').write_text(json.dumps(dict(against_hf=result,against_ep1=cross),indent=2))
print({k:{s:v for s,v in r.items() if s!='records'} for k,r in result.items()})
