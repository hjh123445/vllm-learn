"""Opt-in tiny synthetic CUDA/NCCL capture through production spawned workers."""
import json
import os
from pathlib import Path
import sys
import time
import subprocess
import urllib.request

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_pipeline_worker_runtime import checkpoint, free_port
from engine.core import EngineCore
from samplingParams import SamplingParams
from utils.observation_config import configure_observation
from utils.observed_runner import ObservedModelRunner, ObservedPipelineTransport
from utils.trace_audit import audit_trace

pytestmark=pytest.mark.skipif(os.getenv('PP_TEST_BACKEND')!='nccl',reason='explicit CUDA correctness/profile smoke only')


@pytest.mark.parametrize('pp,tp',[(1,2),(2,2)])
def test_spawned_cuda_trace_and_unprofiled_throughput(tmp_path,monkeypatch,pp,tp):
    monkeypatch.setenv('VLLM_LEARN_REFERENCE_MATH','1')
    monkeypatch.setenv('VLLM_LEARN_ATTENTION_BACKEND','sdpa_reference')
    monkeypatch.setenv('VLLM_LEARN_EXPERT_BACKEND','loop')
    monkeypatch.setenv('VLLM_LEARN_WORKER_TIMEOUT_S','90')
    model=tmp_path/'tiny'
    checkpoint(model)
    outputs=[]
    for profiled in (False,True):
        out=tmp_path/('profile' if profiled else 'bench')
        configure_observation(out,profile=profiled,start_message=2,messages=2)
        core=EngineCore(str(model),tp_size=tp,pp_size=pp,enable_expert_parallel=True,
            master_port=free_port(),gpus=list(range(pp*tp)),
            runner_cls=ObservedModelRunner,transport_cls=ObservedPipelineTransport,
            pipeline_microbatch_tokens=8,config_kwargs=dict(max_model_len=128,
            max_num_batched_tokens=8,max_num_seqs=16,gpu_memory_utilization=.98,
            max_kvcache_blocks=16))
        try:
            started=time.perf_counter()
            result=core.generate([[1,2],[3,4,5],[1,2]],SamplingParams(temperature=0.,max_tokens=4,ignore_eos=True))
            elapsed=time.perf_counter()-started
            outputs.append([tokens for _,tokens in result])
        finally:
            core.shutdown()
        assert all(len(tokens)==4 for tokens in outputs[-1])
        if not profiled:
            assert not list(out.glob('trace/*.json'))
            assert elapsed>0
            (out/'smoke_throughput.json').write_text(json.dumps(dict(output_tokens=12,
                wall_s=elapsed,output_tok_s=12/elapsed,scope='tiny synthetic smoke, not30B/performance result')))
        else:
            traces=sorted((out/'trace').glob('cuda_rank*.json'))
            assert len(traces)==pp*tp
            audits=[]
            for trace in traces:
                audit=audit_trace(trace)
                assert audit['cpu_ops']>0 and audit['cuda_runtime']>0,audit
                assert audit['gpu_kernels']>0 and audit['nccl_kernels']>0,audit
                assert any('.attention' in name for name in audit['module_labels']),audit
                assert any('.moe' in name for name in audit['module_labels']),audit
                audits.append(audit)
            (out/'validation.json').write_text(json.dumps(audits,indent=2))
    assert outputs[0]==outputs[1]  # Profiling observer must not change generated tokens.


def test_real_offline_and_http_command_line(tmp_path,monkeypatch):
    """CLI accepts flags, writes metrics, and profiles a real one-GPU server."""
    monkeypatch.setenv('VLLM_LEARN_REFERENCE_MATH','1')
    monkeypatch.setenv('VLLM_LEARN_ATTENTION_BACKEND','sdpa_reference')
    monkeypatch.setenv('VLLM_LEARN_EXPERT_BACKEND','loop')
    model=tmp_path/'tiny'
    checkpoint(model)
    # server chat entry needs a template, even for our synthetic tokenizer.
    (model/'chat_template.json').write_text(json.dumps(dict(chat_template=
        "{% for message in messages %}t1 t2 {% endfor %}{% if add_generation_prompt %}t3{% endif %}")))
    root=Path(__file__).resolve().parents[1]
    inputs=tmp_path/'inputs.json'
    inputs.write_text(json.dumps(dict(tiny=[dict(input_ids=[1,2],max_tokens=4,arrival_step=0)
                                          for _ in range(8)])))
    for profile in (False,True):
        out=tmp_path/('offline_profile' if profile else 'offline_bench')
        command=[sys.executable,str(root/'bench.py'),'offline','--model',str(model),
            '--output',str(out),'--inputs',str(inputs),'--workloads','tiny','--repeats','1',
            '--max-kv-blocks','16','--profile-start-message','2','--profile-messages','2']
        if profile:command.append('--profile')
        result=subprocess.run(command,capture_output=True,text=True,timeout=60)
        (tmp_path/f'offline_{profile}.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stdout+result.stderr
        records=json.loads((out/'results.json').read_text())
        record=next(r for r in records if r['workload']=='tiny')
        assert record['profiled']==profile
        if profile:
            assert record['output_tok_s'] is None and record['ttft_s'] is None
            assert audit_trace(out/'trace/cuda_rank0.json')['cuda_verified']
        else:
            assert record['output_tok_s']>0 and set(record['ttft_s'])=={'50','95','99'}
    # Real server in profile mode; client should auto-detect via /health and
    # suppress throughput even when --profile is absent from the client.
    port=free_port()
    server_out=tmp_path/'server_profile'
    log=(tmp_path/'server.log').open('w')
    server=subprocess.Popen([sys.executable,str(root/'server.py'),'--model',str(model),
        '--port',str(port),'--host','127.0.0.1','--max-num-seqs','16','--max-model-len','128',
        '--max-num-batched-tokens','16','--max-kvcache-blocks','16','--gpu-mem-util','.98',
        '--profile','--profile-dir',str(server_out),'--profile-start-message','2',
        '--profile-messages','2'],stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+60
        while True:
            if server.poll() is not None:
                raise RuntimeError((tmp_path/'server.log').read_text())
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as response:
                    assert json.load(response)['profiled_run'] is True
                    break
            except OSError:
                if time.monotonic()>deadline:raise
                time.sleep(.1)
        client_out=tmp_path/'http_load'
        result=subprocess.run([sys.executable,str(root/'bench.py'),'serve','--base-url',
            f'http://127.0.0.1:{port}/v1','--output',str(client_out),
            '--requests','8','--concurrency','4','--max-tokens','4','--warmup','0'],
            capture_output=True,text=True,timeout=60)
        (tmp_path/'client.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stdout+result.stderr
        summary=json.loads((client_out/'summary.json').read_text())
        assert summary['successes']==8 and summary['profiled'] is True
        assert summary['output_tok_s'] is None and summary['missing_usage']==0
    finally:
        server.terminate()
        try:server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
        log.close()
    assert server.returncode==0,(tmp_path/'server.log').read_text()
    assert audit_trace(server_out/'trace/cuda_rank0.json')['cuda_verified']
