"""Compare full raw-greedy rollouts at one-at-a-time and batched serving."""
import argparse,json,pathlib,sys,os
sys.path.insert(0,str(pathlib.Path.cwd()/'v9'))

def summary(reference,actual):
 n=min(len(reference),len(actual));first=next((i for i in range(n) if reference[i]!=actual[i]),None)
 return dict(reference_tokens=len(reference),actual_tokens=len(actual),exact=reference==actual,matched_prefix=n if first is None else first,first_mismatch=first)

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);p.add_argument('--ep',type=int,default=8);p.add_argument('--pp',type=int,default=1)
 a=p.parse_args();out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True);refs=json.loads(pathlib.Path(a.reference).read_text())
 from engine.core import EngineCore
 from samplingParams import SamplingParams
 core=EngineCore(a.model,ep_size=a.ep,pp_size=a.pp,master_port=31630,config_kwargs=dict(max_model_len=8192,max_num_batched_tokens=512,max_num_seqs=128,max_kvcache_blocks=256,gpu_memory_utilization=.90,enforce_eager=True,enable_prefix_caching=False))
 results={}
 try:
  single={}
  for r in refs:
   seq=core.add_request(r['input_ids'],SamplingParams(temperature=0.,max_tokens=len(r['output_ids']),ignore_eos=True))
   while not core.is_finished():core.step()
   single[(r['workload'],r['index'])]=list(seq.completion_token_ids)
  batch={};pending=[]
  for r in refs:
   seq=core.add_request(r['input_ids'],SamplingParams(temperature=0.,max_tokens=len(r['output_ids']),ignore_eos=True));pending.append((r,seq))
  while not core.is_finished():core.step()
  for r,seq in pending:batch[(r['workload'],r['index'])]=list(seq.completion_token_ids)
  records=[]
  for r in refs:
   key=(r['workload'],r['index']);h=r['output_ids'];s=single[key];b=batch[key]
   records.append(dict(workload=key[0],index=key[1],hf_vs_custom_single=summary(h,s),hf_vs_custom_batch=summary(h,b),custom_single_vs_batch=summary(s,b),hf_tokens=h,custom_single_tokens=s,custom_batch_tokens=b))
  results=dict(topology=dict(ep=a.ep,pp=a.pp),cases=len(records),exact_hf_single=sum(x['hf_vs_custom_single']['exact'] for x in records),exact_hf_batch=sum(x['hf_vs_custom_batch']['exact'] for x in records),exact_custom_single_vs_batch=sum(x['custom_single_vs_batch']['exact'] for x in records),records=records)
  (out/'rollout_comparison.json').write_text(json.dumps(results,indent=2))
  print(json.dumps({k:v for k,v in results.items() if k!='records'},indent=2),flush=True)
 finally:core.shutdown()
if __name__=='__main__':main()
