import json,re,math
from pathlib import Path
from collections import Counter

src=Path('outputs/verified_parallel/rollout_trace_056_20261104_152543/trace_ep8/trace')
out=Path('outputs/verified_parallel/trace_comparison_056_20261105/custom_ep8_compact_cpu_cuda.json')
out.parent.mkdir(parents=True,exist_ok=True)
all_events=[]; ranks=[]; rank_intervals=[]
layer_re=re.compile(r'stage(\d+)\.layer(?:\.?)(\d+)\.(attention|moe)')
msg_re=re.compile(r'rank(\d+)\.stage(\d+)\.message(\d+)\.mb(\d+)\.(\w+)')
for rank in range(8):
 p=src/f'cuda_rank{rank}.json'
 d=json.loads(p.read_text(encoding='utf-8'))
 ev=d.get('traceEvents',[])
 msg=[e for e in ev if e.get('cat')=='user_annotation' and msg_re.search(str(e.get('name','')))]
 # use the first message on every rank; same workload and complete profile window
 m=sorted(msg,key=lambda e:e.get('ts',0))[0]
 lo=m['ts']-1000; hi=m['ts']+m.get('dur',0)+1000
 gpu_layers=[e for e in ev if e.get('cat')=='gpu_user_annotation' and e.get('ph')=='X' and lo<=e.get('ts',0)<=hi and layer_re.search(str(e.get('name','')))]
 cpu_layers=[e for e in ev if e.get('cat')=='user_annotation' and e.get('ph')=='X' and lo<=e.get('ts',0)<=hi and layer_re.search(str(e.get('name','')))]
 ranges=sorted(gpu_layers+cpu_layers,key=lambda e:e.get('ts',0))
 rank_intervals.append(dict(rank=rank,raw_file=str(p),message=m['name'],start_us=lo,end_us=hi,base_time_ns=d.get('baseTimeNanoseconds'),raw_events=len(ev),selected_ranges=len(ranges)))
 def enclosing_layer(ts):
  best=None
  for e in ranges:
   a=e['ts']; b=a+e.get('dur',0)
   if a<=ts<=b:
    if best is None or e.get('dur',0)<best.get('dur',0): best=e
  if best:
   mm=layer_re.search(str(best.get('name','')))
   if mm: return dict(stage=int(mm.group(1)),layer_global=int(mm.group(2)),op_type=mm.group(3),annotation=best.get('name'))
  return {}
 count=0
 for e in ev:
  ts=e.get('ts')
  if ts is None or not (lo<=ts<=hi): continue
  cat=e.get('cat')
  if cat not in {'user_annotation','cuda_runtime','kernel','gpu_user_annotation','gpu_memcpy','gpu_memset'}: continue
  x=dict(e)
  x.pop('pid',None); x.pop('tid',None)
  args=dict(x.get('args') or {})
  args.update(rank=rank,gpu=rank,source_category=cat)
  layer=enclosing_layer(ts)
  args.update(layer)
  if cat in {'kernel','gpu_user_annotation','gpu_memcpy','gpu_memset'}:
   stream=args.get('stream',e.get('tid',7))
   try: tid=100+int(stream)
   except Exception: tid=100
   if cat=='kernel' and 'nccl' in str(e.get('name','')).lower():
    x['cat']='nccl'; args['op_type']='NCCL'; args.setdefault('collective',args.get('Collective name','unknown'))
   else: x['cat']=cat
   x['pid']=rank; x['tid']=tid
  else:
   x['cat']=cat; x['pid']=rank; x['tid']=10000+rank
  x['args']=args
  # Make important labels visible in the UI without destroying original kernel name.
  if x['cat']=='nccl': x['name']='NCCL | '+str(e.get('name',''))
  all_events.append(x);count+=1
 # metadata process and tracks
 # Keep the track IDs identical to the IDs used above for events.  In
 # particular, rank 7's CPU tid (10007) must not be confused with the CUDA
 # stream tid (107), and the metadata must name tid 107 rather than 10007.
 all_events.extend([
  {'name':'process_name','ph':'M','pid':rank,'args':{'name':f'GPU {rank} / EP rank {rank} / custom EP8'}},
  {'name':'thread_name','ph':'M','pid':rank,'tid':107,'args':{'name':f'CUDA stream 7 | GPU {rank} | kernels + NCCL'}},
  {'name':'thread_sort_index','ph':'M','pid':rank,'tid':107,'args':{'sort_index':0}},
  {'name':'thread_name','ph':'M','pid':rank,'tid':10000+rank,'args':{'name':f'CPU API / annotations | rank {rank}'}},
  {'name':'thread_sort_index','ph':'M','pid':rank,'tid':10000+rank,'args':{'sort_index':-1}},
 ])
summary={"schema":"chrome_trace_event_format","displayTimeUnit":"ms","clock":"torch.profiler/CUPTI timestamps in microseconds; baseTimeNanoseconds preserved per rank","scope":"custom EP8 first message window; CPU annotations + CUDA runtime + GPU stream + NCCL/kernel events","selection":"rank*.stage0.message1.mb0.prefill +/- 1 ms; profile-only, not throughput","ranks":rank_intervals,"events":len(all_events),"categories":dict(Counter(e.get('cat','?') for e in all_events))}
# Sort by timestamp but metadata first is also accepted; sorting makes timeline easier to scan.
all_events.sort(key=lambda e:(0 if e.get('ph')=='M' else 1,e.get('ts',0)))
out.write_text(json.dumps({"displayTimeUnit":"ms","traceEvents":all_events,"metadata":summary},ensure_ascii=False,separators=(',',':')),encoding='utf-8')
(out.with_suffix('.summary.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2)); print('file_bytes',out.stat().st_size)
