"""Run the authorized 128/256 concurrency matrix with isolated logs and PIDs."""
from pathlib import Path
import json
import hashlib
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
MODEL='/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct'
MANIFEST='/data/ifly_jyhe23/vllm-learn-graph-audit-20261108/datasets/chartqa/subset/manifest.json'
OUT=ROOT/'outputs/load_matrix'
SCENARIOS=[('small',16384,0,16),('medium',65536,256,64),('large',262144,1024,64)]
ENV=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',VLLM_LEARN_CPU_THREADS='4')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


class MemorySamples:
    def __init__(self,out):
        self.out=out;self.rows=[];self.done=threading.Event()
    def sample(self):
        try:
            p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10,check=True)
            for line in p.stdout.splitlines():
                index,memory,util=map(int,line.split(','))
                self.rows.append(dict(time=time.time(),gpu=index,memory_mib=memory,utilization_pct=util))
            if 'online' in self.out.name:
                with urllib.request.urlopen('http://127.0.0.1:8018/health',timeout=2) as response:
                    self.rows.append(dict(time=time.time(),engine=json.load(response)))
        except Exception as exc:self.rows.append(dict(error=repr(exc)))
    def loop(self):
        while not self.done.is_set():self.sample();self.done.wait(1.)
    def __enter__(self):
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.done.set();self.thread.join(12);self.sample()
        write(self.out/'gpu_samples.json',self.rows)


def run(command,out,env=ENV):
    out.mkdir(parents=True,exist_ok=True)
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'v9').rglob('*.py') if '__pycache__' not in p.parts}
    if (out/'case.json').exists():
        previous=json.loads((out/'case.json').read_text())
        if (previous.get('status')=='passed' and previous['command']==command
                and previous.get('source_sha256')==source):
            print('CASE_REUSED',out.name,flush=True);return
        raise RuntimeError(f'preserve existing case before rerun: {out}')
    state=dict(command=command,start=time.time(),status='running',source_sha256=source)
    with (out/'run.log').open('w') as log,MemorySamples(out):
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT)
        state['pid']=child.pid;write(out/'case.json',state)
        code=child.wait()
    state.update(status='passed' if code==0 else 'failed',returncode=code,end=time.time())
    write(out/'case.json',state);print('CASE',out.name,state['status'],flush=True)
    if code:raise RuntimeError(f'case failed: {out}')


def client_args(mode,scope,c,pixels,context,output,out):
    return [sys.executable,'-u','v9/tools/evaluate_multimodal.py','--mode',mode,'--model',MODEL,
        '--manifest',MANIFEST,'--scope',scope,'--concurrency',str(c),'--requests',str(2*c),
        '--repeats','3','--max-pixels',str(pixels),'--context-tokens',str(context),
        '--output-tokens',str(output),'--output',str(out)]


def server(scope,out,graph=True):
    out.mkdir(parents=True,exist_ok=True)
    env=dict(ENV,VLLM_LEARN_MM='1' if scope=='multimodal' else '0',
        VLLM_LEARN_BATCH_INVARIANT='1',VLLM_LEARN_REFERENCE_QK='0',
        VLLM_LEARN_ATTENTION_BACKEND='triton',VLLM_LEARN_EP_DENSE_DECODE='1',
        VLLM_LEARN_DENSE_EXPERT_BACKEND='bmm',VLLM_LEARN_EXPERT_BACKEND='torch_grouped',
        VLLM_LEARN_DECODE_GRAPH='1' if graph else '0')
    cmd=[sys.executable,'-u','v9/server.py','--model',MODEL,'--host','127.0.0.1','--port','8018',
        '--tensor-parallel-size','8','--enable-expert-parallel','--max-num-seqs','256',
        '--max-model-len','4096','--max-num-batched-tokens','2048','--max-kvcache-blocks','2048',
        '--gpu-mem-util','.75','--disable-prefix-caching','--graph-reserve-gib','2',
        '--request-timeout','1200','--metrics-path',str(out/'http_requests.jsonl'),
        '--execution-metrics-dir',str(out/'worker'),'--mm-max-pixels','262144']
    if scope=='multimodal':cmd+=['--multimodal']
    log=(out/'run.log').open('w')
    process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    write(out/'server.json',dict(pid=process.pid,command=cmd,start=time.time()))
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError(f'server failed: {out}')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8018/health',timeout=2) as r:
                if json.load(r).get('accepting'):return process,log
        except OSError:pass
        time.sleep(1.)
    process.terminate();process.wait(120);log.close();raise TimeoutError('server startup')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # A smoke run catches engine/HTTP regressions before the sustained matrix.
    smoke=OUT/'smoke_offline'
    run(client_args('offline','multimodal',4,16384,0,4,smoke)+[
        '--graph','--requests','8','--repeats','1','--kv-blocks','64'],smoke)
    process,log=server('multimodal',OUT/'server_smoke')
    try:
        smoke=OUT/'smoke_online_multimodal'
        run(client_args('online','multimodal',4,16384,0,4,smoke)+['--graph','--requests','8','--repeats','1'],smoke)
    finally:
        process.send_signal(signal.SIGTERM);code=process.wait(180);log.close()
        if code:raise RuntimeError('smoke server cleanup failed')
    for scope in ('multimodal','text'):
        for c in (128,256):
            for name,pixels,context,output in SCENARIOS:
                out=OUT/f'offline_{scope}_{name}_c{c}_graph'
                run(client_args('offline',scope,c,pixels,context,output,out)+['--graph'],out)
    # Same numerical policy and dense EP backend: isolate Graph launch cost.
    for c in (128,256):
        out=OUT/f'offline_multimodal_medium_c{c}_eager'
        run(client_args('offline','multimodal',c,65536,256,64,out),out)
    for scope in ('multimodal','text'):
        process,log=server(scope,OUT/f'server_{scope}_graph')
        try:
            smoke=OUT/f'smoke_online_{scope}'
            run(client_args('online',scope,4,16384,0,4,smoke)+['--graph','--requests','8','--repeats','1'],smoke)
            for c in (128,256):
                for name,pixels,context,output in SCENARIOS:
                    out=OUT/f'online_{scope}_{name}_c{c}_graph'
                    run(client_args('online',scope,c,pixels,context,output,out)+['--graph'],out)
        finally:
            process.send_signal(signal.SIGTERM)
            try:code=process.wait(180)
            except subprocess.TimeoutExpired:process.kill();code=process.wait()
            log.close()
            write(OUT/f'server_{scope}_graph/exit.json',dict(returncode=code,end=time.time()))
            if code:raise RuntimeError('server did not cleanly shutdown')
    print('MATRIX_PASSED',flush=True)


if __name__=='__main__':main()
