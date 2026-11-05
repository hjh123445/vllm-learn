import pathlib,json,socket,collections
assert socket.gethostname()=='task3-A800-056'
p=pathlib.Path('/data/ifly_jyhe23/hot_model_top5/audio_understand/qwen-audio/vllm-learn/outputs/verified_parallel/rollout_trace_056_20261104_152543')
for name in ['trace_ep8','trace_pp2ep4_baseline','trace_pp2ep4_dedicated']:
 t=p/name/'trace';print('\n',name)
 for rank in range(8):
  f=t/f'cuda_rank{rank}.json';capture=t/f'capture_rank{rank}.json'
  if not f.exists():print('MISSING',rank);continue
  d=json.loads(f.read_text());es=d['traceEvents'];k=[e for e in es if e.get('cat')=='kernel'];layers=[e for e in es if e.get('cat')=='gpu_user_annotation']
  prof=json.loads(capture.read_text()) if capture.exists() else {}
  ann=collections.Counter(e.get('name') for e in layers)
  print(rank,'kernel',len(k),'layer_annotations',len(layers),'distinct_layers',len(ann),'capture',prof.get('captured_messages'))
