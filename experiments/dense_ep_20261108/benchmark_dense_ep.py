"""Controlled full-model decode comparison; no production source modifications.

Uses real TP8 weights, identical prefilled KV/input IDs per backend, fixed
sequence positions, and includes runner metadata preparation and sampling.
This is a repeated decode-step benchmark, not an online request benchmark.
"""
import argparse,json,os,sys,time,hashlib,statistics
from pathlib import Path

def worker(rank,args):
    import torch
    import torch.distributed as dist
    import faulthandler
    faulthandler.dump_traceback_later(900)
    torch.set_num_threads(4);torch.cuda.set_device(rank)
    sys.path.insert(0,str(Path(args.source)/'v9'))
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{args.port}',rank=rank,
        world_size=8,device_id=torch.device(f'cuda:{rank}'))
    from engine.process_groups import create_groups
    from layers.parallel import configure_parallel
    groups=create_groups(rank,tp_size=8,enable_expert_parallel=True)
    for g in groups.values():
        if g is not None:dist.all_reduce(torch.zeros(1,device='cuda'),group=g)
    configure_parallel(tp_size=8,tp_rank=rank,tp_group=groups['tp'],ep_group=groups['ep'],enable_expert_parallel=True)
    from config import Config
    from engine.model_runner import ModelRunner
    from engine.sequence import Sequence
    from samplingParams import SamplingParams
    from layers.moe import MoEMLP
    cfg=Config(args.model,max_model_len=256,max_num_batched_tokens=2048,max_num_seqs=256,
        max_kvcache_blocks=256,decode_graph_reserve_bytes=2*1024**3,gpu_memory_utilization=.8)
    runner=ModelRunner(cfg)
    graphs=runner.decode_graphs
    moes=[m for m in runner.model.modules() if isinstance(m,MoEMLP)]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    def mode(name):
        graph=name=='dense_graph_strict';dense=name.startswith('dense_')
        os.environ['VLLM_LEARN_BATCH_INVARIANT']='0' if name=='sparse_eager_native' else '1'
        for m in moes:m.ep_dense_decode=dense
        runner.decode_graphs=graphs if graph else {}
        runner._decode_graph_requested=graph
    try:
        with torch.inference_mode():
            for b in args.batches:
                mode('dense_graph_strict')
                rng=torch.Generator().manual_seed(20261108)
                seqs=[]
                for i in range(b):
                    s=Sequence(torch.randint(100,30000,(args.prefix,),generator=rng).tolist(),SamplingParams(temperature=0.))
                    s.block_table=[i];s.num_scheduled_tokens=args.prefix;seqs.append(s)
                for start in range(0,b,max(1,2048//args.prefix)):
                    runner.run(seqs[start:start+max(1,2048//args.prefix)],True)
                for s in seqs:s.append_token(1234);s.num_scheduled_tokens=1
                before=runner.kv_cache.clone()
                names=['sparse_eager_native','sparse_eager_strict','dense_eager_strict','dense_graph_strict']
                for repeat in range(args.repeats):
                    order=names[repeat%len(names):]+names[:repeat%len(names)]
                    for name in order:
                        mode(name);runner.kv_cache.copy_(before)
                        for _ in range(2):runner.run(seqs,False)
                        torch.cuda.synchronize();dist.barrier()
                        start=time.perf_counter()
                        for _ in range(args.iterations):tokens=runner.run(seqs,False)
                        torch.cuda.synchronize()
                        elapsed=torch.tensor(time.perf_counter()-start,device='cuda',dtype=torch.float64)
                        dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
                        seconds=elapsed.item()/args.iterations
                        row=dict(batch=b,prefix=args.prefix,mode=name,repeat=repeat,
                            step_ms=seconds*1000,decode_output_tok_s=b/seconds,iterations=args.iterations)
                        if rank==0:
                            rows.append(row)
                            (out/'progress.json').write_text(json.dumps(rows,indent=2))
                            print(json.dumps(row),flush=True)
                del before
            if rank==0:
                summary=[]
                for b in args.batches:
                    record=dict(batch=b,prefix=args.prefix)
                    for name in names:
                        ms=statistics.median(r['step_ms'] for r in rows if r['batch']==b and r['mode']==name)
                        record[name]=dict(step_ms=ms,decode_output_tok_s=b*1000/ms)
                    record['dense_graph_vs_sparse_native_speedup']=record['sparse_eager_native']['step_ms']/record['dense_graph_strict']['step_ms']
                    record['dense_graph_vs_sparse_strict_speedup']=record['sparse_eager_strict']['step_ms']/record['dense_graph_strict']['step_ms']
                    summary.append(record)
                result=dict(measurement='real TP8+EP full-model repeated decode; fixed KV/input positions; preparation+sampling included; no prefill/HTTP; max elapsed across ranks',
                    model=args.model,options=vars(args),rows=rows,summary=summary,
                    numerical_policy='native sparse is a separate arithmetic baseline; strict sparse/dense share fixed primitives but different route reduction order; no cross-backend bitwise equality assumed',
                    source_sha256={str(p.relative_to(Path(args.source))):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(args.source)/'v9').rglob('*.py') if '__pycache__' not in p.parts})
                (out/'results.json').write_text(json.dumps(result,indent=2))
    finally:
        runner.decode_graphs=graphs;runner.shutdown()
        dist.destroy_process_group();faulthandler.cancel_dump_traceback_later()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--port',type=int,default=31959);p.add_argument('--prefix',type=int,default=128)
    p.add_argument('--batches',type=int,nargs='+',default=[1,8,32,128,256])
    p.add_argument('--iterations',type=int,default=6);p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if not 1<=args.prefix<256:p.error('prefix must fit one KV block including decode')
    os.environ.update(VLLM_LEARN_DECODE_GRAPH='1',VLLM_LEARN_EP_DENSE_DECODE='1',
        VLLM_LEARN_BATCH_INVARIANT='1',VLLM_LEARN_REFERENCE_QK='0',VLLM_LEARN_ATTENTION_BACKEND='triton',
        VLLM_LEARN_EXPERT_BACKEND='torch_grouped',VLLM_LEARN_DENSE_EXPERT_BACKEND='bmm',VLLM_LEARN_MM='0',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    import torch.multiprocessing as mp
    mp.spawn(worker,args=(args,),nprocs=8,join=True)
