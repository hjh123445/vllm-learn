"""CLI and real loopback HTTP checks; no model download or GPU."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from server import Handler
from utils.http_benchmark import request, summarize, server_profile_status
from utils.observation_config import configure_observation
from utils.trace_audit import audit_trace


class FakeEngine:
    def stats(self):
        return dict(accepting=True)

    def add(self, ids, sampling):
        stream=queue.Queue()
        for token in (0,1,2,3):
            stream.put(token)
        stream.put(None)
        return 1,SimpleNamespace(q=stream,error=None,finish_reason='length',times={})

    def abort(self, _):
        pass


@pytest.fixture
def http_server():
    class LocalHandler(Handler):
        engine=FakeEngine()
        tokenizer=SimpleNamespace(apply_chat_template=lambda *_,**__: 'prompt',
            encode=lambda *_,**__: [1,2],
            decode=lambda ids,**_: ''.join('' if t==0 else str(t) for t in ids))
        metrics_path=None
    srv=ThreadingHTTPServer(('127.0.0.1',0),LocalHandler)
    worker=threading.Thread(target=srv.serve_forever,daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{srv.server_port}/v1'
    finally:
        srv.shutdown()
        srv.server_close()
        worker.join()


def test_sse_counts_usage_and_ignores_role_only_or_empty_chunks(http_server):
    assert server_profile_status(http_server,5) is False
    payload=dict(messages=[dict(role='user',content='合成测试')],stream=True,
                 stream_options=dict(include_usage=True),max_tokens=4,temperature=0.)
    row=request(http_server,payload,0,5)
    assert row['success'] and row['http_status']==200
    assert row['output_tokens']==4
    assert len(row['content_chunks'])==3  # First token is invisible; never count chunks as tokens.
    summary=summarize([row],1.,1,False)
    assert summary['output_tok_s']==4 and summary['request_s']==1
    assert summary['e2e_s']['99']>=summary['ttft_content_s']['99']>=0
    assert summary['failures']==0
    diagnostic=summarize([row],1.,1,True)
    assert diagnostic['benchmark_valid'] is False and diagnostic['output_tok_s'] is None


def test_missing_usage_does_not_invent_tokens():
    row=dict(success=True,output_tokens=None,e2e_s=.5,ttft_content_s=.1,
             tpot_content_s=None,chunk_itl_s=[.1])
    summary=summarize([row],1.,1,False)
    assert summary['missing_usage']==1 and summary['output_tok_s'] is None
    assert summary['request_s']==1


def test_bench_http_cli_saves_requests_percentiles_and_errors(http_server,tmp_path):
    cli=Path(__file__).resolve().parents[1]/'bench.py'
    command=[sys.executable,str(cli),'serve','--base-url',http_server,'--output',str(tmp_path),
             '--requests','8','--concurrency','4','--warmup','1','--max-tokens','4']
    result=subprocess.run(command,text=True,capture_output=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['successes']==8 and summary['output_tokens']==32
    assert set(summary['e2e_s'])=={'50','95','99'}
    assert len(json.loads((tmp_path/'requests.json').read_text()))==8


@pytest.mark.parametrize('mode',['offline','serve','trace-check'])
def test_cli_help(mode):
    cli=Path(__file__).resolve().parents[1]/'bench.py'
    result=subprocess.run([sys.executable,str(cli),mode,'--help'],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert '--profile' in result.stdout if mode!='trace-check' else '--require-cuda' in result.stdout


def test_benchmark_configuration_clears_inherited_diagnostics(monkeypatch,tmp_path):
    monkeypatch.setenv('VLLM_LEARN_CUDA_PROFILE','1')
    monkeypatch.setenv('VLLM_LEARN_LOGITS_DIR','unwanted')
    monkeypatch.setenv('VLLM_LEARN_TRACE_DIR','old')
    monkeypatch.setenv('VLLM_LEARN_PROFILE_MESSAGES','999')
    configure_observation(tmp_path)
    import os
    assert 'VLLM_LEARN_TRACE_DIR' not in os.environ
    assert 'VLLM_LEARN_LOGITS_DIR' not in os.environ
    configure_observation(tmp_path,profile=True,start_message=2,messages=3)
    assert os.environ['VLLM_LEARN_PROFILE_MESSAGES']=='3'


def test_trace_audit_requires_actual_gpu_events(tmp_path):
    path=tmp_path/'cuda_rank0.json'
    path.write_text(json.dumps(dict(traceEvents=[dict(name='stage0.layer0',cat='user_annotation',ph='X')],)),encoding='utf-8')
    audit=audit_trace(path)
    assert not audit['cuda_verified'] and audit['gpu_kernels']==0
    assert audit['module_labels']==['stage0.layer0']
