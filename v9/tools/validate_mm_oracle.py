"""Teacher-forced TP8/EP Graph comparison with actual HF generation logits."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def worker(rank,args):
    import torch
    import torch.distributed as dist
    from engine.process_groups import create_groups
    from layers.parallel import configure_parallel
    torch.set_num_threads(2);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{args.port}',rank=rank,
        world_size=args.tp,device_id=torch.device(f'cuda:{rank}'))
    groups=create_groups(rank,tp_size=args.tp,enable_expert_parallel=True)
    for group in groups.values():
        if group is not None:dist.all_reduce(torch.zeros(1,device='cuda'),group=group)
    configure_parallel(tp_size=args.tp,tp_rank=rank,tp_group=groups['tp'],
        ep_group=groups['ep'],enable_expert_parallel=True)
    from config import Config
    from engine.model_runner import ModelRunner
    from engine.sequence import Sequence
    from samplingParams import SamplingParams
    oracle=torch.load(args.oracle,map_location='cpu',weights_only=False)
    runner=ModelRunner(Config(args.model,max_model_len=4096,max_num_batched_tokens=2048,
        max_num_seqs=1,max_kvcache_blocks=32,gpu_memory_utilization=.75))
    rows=[];captured=[]
    handle=runner.sampler.register_forward_pre_hook(lambda _m,inputs:captured.append(inputs[0].float().cpu()))
    try:
        image=oracle['mm']['images'][0]
        with torch.inference_mode():
            merged,deep=runner.model.run_vision(torch.from_numpy(image['pixels']).cuda(),
                                              torch.tensor([image['grid']]))
        vision=[]
        for name,a,b in [('merged',merged,oracle['merged'])]+[
                (f'deepstack{i}',a,b) for i,(a,b) in enumerate(zip(deep,oracle['deepstack']))]:
            a=a.float().cpu();d=(a-b).abs()
            vision.append(dict(component=name,max_abs=d.max().item(),mean_abs=d.mean().item(),exact=torch.equal(a,b)))
        seq=Sequence(oracle['input_ids'],SamplingParams(temperature=0.))
        seq.mm=oracle['mm'];seq.block_table=list(range(seq.num_blocks));seq.num_scheduled_tokens=len(seq)
        for step,reference in enumerate(oracle['step_logits']):
            if step:
                seq.append_token(oracle['output_ids'][step-1]);seq.num_scheduled_tokens=1
                if seq.num_blocks>len(seq.block_table):seq.block_table.append(len(seq.block_table))
            runner.run([seq],step==0)
            actual=captured.pop();ref=reference.float();diff=(actual-ref).abs()
            graph_eager_exact=kv_graph_eager_equal=None
            if step:
                with torch.inference_mode():
                    kv_graph=runner.kv_cache.clone()
                    ids,pos=runner.prepare_decode([seq])
                    eager=runner.model.compute_logits(runner.model(ids,pos)).float().cpu()
                    graph_eager_exact=torch.equal(actual,eager)
                    kv_graph_eager_equal=torch.equal(kv_graph,runner.kv_cache)
                    del kv_graph
                if not (graph_eager_exact and kv_graph_eager_equal):
                    raise AssertionError('real multimodal prefix Graph/eager logits or KV differ')
            hf_top=torch.topk(ref[0],5);own_top=torch.topk(actual[0],5)
            rows.append(dict(step=step,max_abs=diff.max().item(),mean_abs=diff.mean().item(),
                graph_eager_exact=graph_eager_exact,kv_graph_eager_equal=kv_graph_eager_equal,
                argmax_equal=actual.argmax(-1).item()==ref.argmax(-1).item(),
                hf_top_ids=hf_top.indices.tolist(),hf_top_logits=hf_top.values.tolist(),
                own_top_ids=own_top.indices.tolist(),own_top_logits=own_top.values.tolist(),
                hf_margin=float(hf_top.values[0]-hf_top.values[1])))
        result=dict(rank=rank,index=oracle['index'],vision=vision,steps=rows,
            graph_replays=runner.graph_replay_count,measurement='diagnostic; same HF prefixes, actual HF generation logits')
        (Path(args.output)/f'rank{rank}.json').write_text(json.dumps(result,indent=2))
        print('ORACLE_DONE',rank,vision,[(r['step'],r['argmax_equal'],r['hf_margin'],r['max_abs']) for r in rows],flush=True)
    finally:
        handle.remove();runner.shutdown();dist.destroy_process_group()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True);ap.add_argument('--oracle',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--tp',type=int,default=8)
    ap.add_argument('--port',type=int,default=31939)
    args=ap.parse_args();Path(args.output).mkdir(parents=True,exist_ok=True)
    os.environ.update(VLLM_LEARN_MM='1',VLLM_LEARN_BATCH_INVARIANT='1',VLLM_LEARN_REFERENCE_QK='0',
        VLLM_LEARN_ATTENTION_BACKEND='triton',VLLM_LEARN_EP_DENSE_DECODE='0',VLLM_LEARN_EP_SPARSE_DECODE='1',
        VLLM_LEARN_DENSE_EXPERT_BACKEND='bmm',VLLM_LEARN_EXPERT_BACKEND='torch_grouped',VLLM_LEARN_DECODE_GRAPH='1')
    import torch.multiprocessing as mp
    mp.spawn(worker,args=(args,),nprocs=args.tp,join=True)


if __name__=='__main__':main()
