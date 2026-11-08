"""Isolate Graph effects on all 256 real-image quality outputs."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from run_load_matrix import MemorySamples

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/completion_suite/quality_offline_eager'
MODEL='/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct'
MANIFEST='/data/ifly_jyhe23/vllm-learn-graph-audit-20261108/datasets/chartqa/subset/manifest.json'

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-u','v9/tools/evaluate_multimodal.py','--mode','offline',
        '--model',MODEL,'--manifest',MANIFEST,'--samples','256','--max-pixels','262144',
        '--output-tokens','64','--output',str(OUT),'--quality','--concurrency','128','--repeats','1']
    state=dict(command=command,start=time.time(),status='running')
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',VLLM_LEARN_CPU_THREADS='4')
    with (OUT/'run.log').open('w') as log,MemorySamples(OUT):
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['pid']=child.pid;(OUT/'case.json').write_text(json.dumps(state,indent=2))
        code=child.wait()
    state.update(status='passed' if code==0 else 'failed',returncode=code,end=time.time())
    (OUT/'case.json').write_text(json.dumps(state,indent=2))
    if code:raise SystemExit(code)
    graph=json.loads((OUT.parent/'quality_offline/results.json').read_text())[0]
    eager=json.loads((OUT/'results.json').read_text())[0]
    refs={q['index']:q for q in graph['requests']}
    differences=[]
    for q in eager['requests']:
        ref=refs[q['index']]
        if ref['output_ids']!=q['output_ids']:
            differences.append(dict(index=q['index'],graph=ref['answer'],eager=q['answer']))
    result=dict(samples=256,exact_token_matches=256-len(differences),differences=differences,
        graph_accuracy=graph['quality_accuracy'],eager_accuracy=eager['quality_accuracy'],
        same_numerical_policy=True,measurement='EOS-stopped ChartQA outputs; separate processes, TP8+EP, same fixed numerical policy')
    (OUT.parent/'graph_eager_quality_comparison.json').write_text(json.dumps(result,indent=2))
    assert not differences,result
    print('GRAPH_EAGER_QUALITY_PASSED',result,flush=True)
    release=ROOT/'outputs/completion_suite/strict_large_release'
    command=[sys.executable,'-u','v9/tools/validate_decode_graph.py','--model',MODEL,
        '--output',str(release),'--max-batch','256','--batches','1','8','9','16','17','32',
        '33','64','65','127','128','129','255','256','--max-model-len','2048',
        '--lengths','1023','1024','1535','2044',
        '--steps','4','--seed','1949','--random-inputs','--permute','--mixed-lengths',
        '--strict-batch-invariant','--batch-invariant','--graph-reserve-gib','2']
    release.mkdir(parents=True,exist_ok=False)
    release_state=dict(command=command,status='running',started=time.time())
    with (release/'run.log').open('w') as log,MemorySamples(release):
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        release_state['pid']=child.pid
        (release/'case.json').write_text(json.dumps(release_state,indent=2))
        code=child.wait()
    release_state.update(status='passed' if code==0 else 'failed',ended=time.time(),returncode=code)
    (release/'case.json').write_text(json.dumps(release_state,indent=2))
    if code:raise SystemExit(code)
    print('RELEASE_GRAPH_REGRESSION_PASSED',flush=True)
