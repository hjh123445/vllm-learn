"""Finish queued checks after the existing isolated GPU load matrix exits."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs/completion_suite'
ENV = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', VLLM_LEARN_CPU_THREADS='4')


def run(script, name, env=ENV):
    path = OUT/name
    path.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', started=time.time(), command=[sys.executable,'-u',script])
    (path/'case.json').write_text(json.dumps(state,indent=2))
    with (path/'run.log').open('w') as log:
        child = subprocess.Popen(state['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['pid']=child.pid
        (path/'case.json').write_text(json.dumps(state,indent=2))
        code=child.wait()
    state.update(status='passed' if code==0 else 'failed', ended=time.time(), returncode=code)
    (path/'case.json').write_text(json.dumps(state,indent=2))
    if code:
        raise RuntimeError(f'{name} failed')
    print('POST_COMPLETE',name,flush=True)


def main():
    state_path=OUT/'resume_job.json'
    print('WAITING_FOR_EXISTING_SUITE',flush=True)
    while True:
        state=json.loads(state_path.read_text())
        if state['status']=='passed':
            break
        if state['status']=='failed':
            raise RuntimeError('existing matrix failed; preserve its evidence')
        time.sleep(10)
    run('run_quality_eager.py','graph_eager_quality_and_release')
    deps='/data/ifly_jyhe23/vllm-learn-graph-audit-20261108/runtime_deps'
    run('validate_real_vision.py','vision_real_job',dict(ENV,PYTHONPATH=deps))
    run('audit_outputs.py','raw_token_audit')
    audit=json.loads((ROOT/'outputs/raw_output_audit.json').read_text())
    assert not audit['cross_case_differences'], audit['cross_case_differences'][:10]
    for check in audit['checks']:
        assert not any(check[key] for key in ('duplicate_input_differences','repeat_differences',
            'sse_content_differences','missing_token_evidence')),check
    cases=list((ROOT/'outputs/load_matrix').glob('*/case.json'))
    regular=[p for p in cases if p.parent.name.startswith(('online_','offline_'))]
    assert len(regular)==26, len(regular)
    for p in regular:
        assert json.loads(p.read_text())['status']=='passed',str(p)
        rounds=json.loads((p.parent/'results.json').read_text())
        assert len(rounds)==3 and all(r['errors']==0 for r in rounds),str(p)
    for folder in [p.parent for p in regular if p.parent.name.startswith('offline_')] + [
            ROOT/'outputs/load_matrix'/f'server_{scope}_graph/worker' for scope in ('text','multimodal')]:
        ranks=list(folder.glob('execution_rank*.json'))
        assert len(ranks)==8, str(folder)
        for p in ranks:
            d=json.loads(p.read_text())
            if d['graph_requested']:
                assert d['graph_replays']>0 and d['captured_buckets'] and d['graph_strict'],str(p)
    summary=dict(passed=True,cases=len(regular),rounds=sum(len(json.loads((p.parent/'results.json').read_text())) for p in regular),
        requests=sum(r['completed'] for p in regular for r in json.loads((p.parent/'results.json').read_text())),
        cross_case_compared=audit['cross_case_compared'],cross_case_differences=0,
        quality=json.loads((OUT/'graph_eager_quality_comparison.json').read_text()))
    (OUT/'acceptance.json').write_text(json.dumps(summary,indent=2))
    print('FINAL_ACCEPTANCE_PASSED',summary,flush=True)


if __name__=='__main__':
    main()
