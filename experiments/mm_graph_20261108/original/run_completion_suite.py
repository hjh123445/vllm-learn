"""Complete source-matched correctness, quality, and load verification."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from run_load_matrix import MemorySamples

ROOT=Path(__file__).resolve().parent
MODEL='/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct'
MANIFEST='/data/ifly_jyhe23/vllm-learn-graph-audit-20261108/datasets/chartqa/subset/manifest.json'
OUT=ROOT/'outputs/completion_suite'
ENV=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',VLLM_LEARN_CPU_THREADS='4')

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)

def run(args,name,env=ENV):
    out=OUT/name;out.mkdir(parents=True,exist_ok=True)
    state=dict(command=[sys.executable,'-u']+args,status='running',started=time.time())
    if name=='quality_hf' and (out/'case.json').exists():
        previous=json.loads((out/'case.json').read_text())
        assert previous['status']=='passed' and previous['command']==state['command']
        print('REUSED_HF_REFERENCE',flush=True);return
    with (out/'run.log').open('w') as log,MemorySamples(out):
        child=subprocess.Popen(state['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['pid']=child.pid;write(out/'case.json',state)
        code=child.wait()
    state.update(status='passed' if code==0 else 'failed',returncode=code,ended=time.time());write(out/'case.json',state)
    if code:raise RuntimeError(f'failed: {name}')
    print('COMPLETE',name,flush=True)

if __name__=='__main__':
    prerequisite=json.loads((ROOT/'outputs/revision_validation_wire/job.json').read_text())
    assert prerequisite['status']=='passed',prerequisite
    for mode in ('hf','offline'):
        name='quality_'+mode
        command=['v9/tools/evaluate_multimodal.py','--mode',mode,'--model',MODEL,
            '--manifest',MANIFEST,'--samples','256','--max-pixels','262144','--output-tokens','64','--output',str(OUT/name)]
        if mode=='offline':command+=['--quality','--graph','--concurrency','128','--repeats','1']
        env=dict(ENV,PYTHONPATH='/data/ifly_jyhe23/vllm-learn-graph-audit-20261108/runtime_deps') if mode=='hf' else ENV
        run(command,name,env)
    hf=json.loads((OUT/'quality_hf/quality.json').read_text())
    own=json.loads((OUT/'quality_offline/results.json').read_text())[0]
    reference={q['index']:q for q in hf['samples']}
    disagreements=[]
    for q in own['requests']:
        ref=reference[q['index']]
        if ref['output_ids']!=q['output_ids']:
            disagreements.append(dict(index=q['index'],hf=ref['answer'],own=q['answer'],
                hf_correct=ref['correct'],own_correct=q['correct']))
    write(OUT/'quality_comparison.json',dict(samples=len(reference),hf_accuracy=hf['relaxed_accuracy'],
        own_accuracy=own['quality_accuracy'],exact_token_matches=len(reference)-len(disagreements),
        disagreements=disagreements,measurement='same ChartQA inputs, EOS stopping, max64 new tokens; HF single-GPU vs TP8+EP strict policy'))
    run(['v9/tools/validate_decode_graph.py','--model',MODEL,'--output',str(OUT/'strict_large'),
        '--max-batch','256','--batches','1','8','9','16','17','32','33','64','65','127','128','129','255','256',
        '--lengths','17','255','256','511','--steps','4','--seed','1949','--random-inputs',
        '--permute','--mixed-lengths','--strict-batch-invariant','--batch-invariant','--graph-reserve-gib','2'], 'strict_large')
    run(['run_load_matrix.py'],'load_matrix')
    run(['v9/tools/summarize_mm_matrix.py','--root',str(ROOT/'outputs/load_matrix'),
         '--output',str(ROOT/'outputs/summary')],'summarize')
    print('COMPLETION_SUITE_PASSED',flush=True)
