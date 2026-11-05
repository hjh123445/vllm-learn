import json,pathlib,sys,torch
from analyze_numerical_diagnostics import compare
root=pathlib.Path(sys.argv[1]);torch.set_num_threads(4);captured={};result={}
for folder in root.iterdir():
 if (folder/'text_reference.json').exists():
  values={}
  for r in json.loads((folder/'text_reference.json').read_text()):
   for i,z in enumerate(torch.load(r['raw_logits_file'],weights_only=True)):values[(r['workload'],r['index'],i)]=z
  captured[folder.name]=values
 elif (folder/'requests.json').exists():
  byseq={}
  for f in sorted((folder/'captures').glob('*.pt')):
   b=torch.load(f,weights_only=True)
   for sid,z in zip(b['seq_ids'],b['logits']):byseq.setdefault(sid,[]).append(z)
  captured[folder.name]={(r['workload'],r['index'],r['prefix_tokens']):byseq[r['seq_id']][r['prefix_tokens']] for r in json.loads((folder/'requests.json').read_text())}
for left,right in [('hf_single','hf_batch'),('hf_single','hf_batch_split_sdpa'),('hf_single','cached_teacher_ep8_single'),('hf_batch','cached_teacher_ep8_batch'),('hf_batch_split_sdpa','cached_teacher_ep8_batch'),('cached_teacher_ep8_batch','cached_teacher_pp2ep4_batch')]:
 keys=sorted(captured[left]);assert set(keys)==set(captured[right]);rows=[dict(workload=k[0],index=k[1],prefix_tokens=k[2],**compare(captured[right][k],captured[left][k])) for k in keys]
 summary=dict(cases=len(rows),matches=sum(r['top1_matches'] for r in rows),exact=sum(r['max_abs_error']==0 for r in rows),worst_abs=max(r['max_abs_error'] for r in rows));print(left,right,summary,flush=True);result[left+'__'+right]=dict(summary,records=rows)
(root/'batch_control_diagnostics.json').write_text(json.dumps(result,indent=2))
