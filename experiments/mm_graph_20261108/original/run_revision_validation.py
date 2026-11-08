from pathlib import Path
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
MODEL='/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct'
ENV=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',VLLM_LEARN_CPU_THREADS='4')
OUT=ROOT/'outputs/revision_validation_wire'

def run(command,name):
    out=OUT/name
    if False:
        print('VALIDATION_ALREADY_PASSED',name,flush=True);return
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.log').open('w') as log:
        subprocess.run([sys.executable,'-u']+command,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('VALIDATION_PASSED',name,flush=True)

if __name__=='__main__':
    for graph in (True,False):
        name='mixed_graph' if graph else 'mixed_eager'
        run(['v9/tools/validate_mixed_decode.py','--model',MODEL,'--output',str(OUT/name)]+(['--graph'] if graph else []),name)
    name='text_small_c256_graph'
    run(['v9/tools/evaluate_multimodal.py','--mode','offline','--model',MODEL,
        '--manifest','/data/ifly_jyhe23/vllm-learn-graph-audit-20261108/datasets/chartqa/subset/manifest.json',
        '--scope','text','--concurrency','256','--requests','512','--repeats','3',
        '--max-pixels','16384','--context-tokens','0','--output-tokens','16','--graph','--output',str(OUT/name)],name)
    reps=json.loads((OUT/name/'results.json').read_text());base={};report=[]
    for rep in reps:
        seen={};within=[];cross=[]
        for q in rep['requests']:
            k=q['index'];ids=q['output_ids'];ordinal=q['request_index']
            if k in seen and seen[k]!=ids:within.append(k)
            seen.setdefault(k,ids)
            if rep['repeat'] and base[ordinal]!=ids:cross.append(ordinal)
            if rep['repeat']==0:base[ordinal]=ids
        report.append(dict(repeat=rep['repeat'],duplicate_input_differences=within,repeat_differences=cross))
    (OUT/'output_consistency.json').write_text(json.dumps(report,indent=2))
    assert all(not r['duplicate_input_differences'] and not r['repeat_differences'] for r in report),report
    print('REVISION_VALIDATION_PASSED',flush=True)

