"""Controlled padding diagnosis. Hooks/copies live only in this probe."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(rank, args):
    import torch
    import torch.distributed as dist
    from config import Config
    from engine.model_runner import ModelRunner
    from engine.sequence import Sequence
    from engine.process_groups import create_groups
    from layers.parallel import configure_parallel
    from samplingParams import SamplingParams
    from utils.context import set_context
    torch.set_num_threads(2); torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{args.port}', rank=rank,
                            world_size=8, device_id=torch.device(f'cuda:{rank}'))
    groups = create_groups(rank, tp_size=8, enable_expert_parallel=True)
    for group in groups.values():
        if group is not None: dist.all_reduce(torch.zeros(1, device='cuda'), group=group)
    configure_parallel(tp_size=8, tp_rank=rank, tp_group=groups['tp'], ep_group=groups['ep'], enable_expert_parallel=True)
    runner = ModelRunner(Config(args.model, max_model_len=512, max_num_batched_tokens=512,
        max_num_seqs=args.bucket, max_kvcache_blocks=16, gpu_memory_utilization=.8))
    seqs = []
    for i in range(args.batch):
        seq = Sequence([i+1]*255, SamplingParams(temperature=0.))
        seq.block_table=[i*2]; seq.num_scheduled_tokens=255; seqs.append(seq)
    runner.run(seqs, True)
    for seq in seqs: seq.append_token(3); seq.num_scheduled_tokens=1
    records, outputs, handles, current = {}, {}, [], [None]
    if args.expert_intermediates:
        import layers.moe as moe_module
        active_layer, active_size = [0], [args.batch]
        for i, layer in enumerate(runner.model.model.layers):
            def select_layer(_module, _inputs, layer_id=i):
                active_layer[0] = layer_id
            handles.append(layer.mlp.register_forward_pre_hook(select_layer))
        for function_name in ('forward_linear', 'silu_and_mul', 'expert_sum',
                              'fixed_all_reduce', '_local_route_weights'):
            original = getattr(moe_module, function_name)
            def instrument(*inputs, function=original, name=function_name, **kwargs):
                output = function(*inputs, **kwargs)
                if current[0] is not None:
                    value = output
                    if name == '_local_route_weights': value = value[:, :args.batch]
                    elif name == 'silu_and_mul':
                        value = value.view(-1, active_size[0], value.shape[-1])[:, :args.batch]
                    elif value.ndim == 3: value = value[:, :args.batch]
                    else: value = value[:args.batch]
                    label = f'layer{active_layer[0]}.expert_{name}'
                    base, serial = label, 1
                    while label in records[current[0]]:
                        serial += 1
                        label = f'{base}_{serial}'
                    source = inputs[0]
                    source = source[:, :args.batch] if source.ndim == 3 else source[:args.batch]
                    records[current[0]][label] = dict(inputs=source.detach().clone(), output=value.detach().clone())
                return output
            setattr(moe_module, function_name, instrument)
    def attach(module, name):
        def hook(_module, inputs, output):
            if current[0] is None: return
            def clone(x):
                if torch.is_tensor(x): return x[:args.batch].detach().clone()
                if isinstance(x, (tuple,list)): return [clone(v) for v in x]
                return None
            records[current[0]][name] = dict(inputs=clone(inputs), output=clone(output))
        handles.append(module.register_forward_hook(hook))
    attach(runner.model.model.embed_tokens, 'embedding')
    for i, layer in enumerate(runner.model.model.layers):
        for suffix, mod in [('q_proj',layer.self_attn.q_proj),('k_proj',layer.self_attn.k_proj),
            ('v_proj',layer.self_attn.v_proj),('paged_attention',layer.self_attn.attn),
            ('o_proj',layer.self_attn.o_proj),('attention',layer.self_attn),
            ('router',layer.mlp.router),('moe',layer.mlp),('layer',layer)]:
            attach(mod, f'layer{i}.{suffix}')
    attach(runner.model.lm_head, 'lm_head')
    with torch.inference_mode():
        snapshot=runner.kv_cache.clone()
        for name, size, dummy in [('eager_unpadded',args.batch,0),('eager_padded',args.bucket,0),('eager_padded_dummy_changed',args.bucket,17),('eager_unpadded_repeat',args.batch,0)]:
            if args.expert_intermediates: active_size[0] = size
            runner.kv_cache.copy_(snapshot)
            ids,pos=runner.prepare_decode(seqs)
            from utils.context import get_context
            ctx=get_context()
            if size > args.batch:
                ids=torch.cat([ids,ids.new_full((size-args.batch,), dummy)]);pos=torch.cat([pos,pos.new_zeros(size-args.batch)])
                slots=torch.cat([ctx.slot_mapping,ctx.slot_mapping.new_full((size-args.batch,), -1)])
                lens=torch.cat([ctx.context_lens,ctx.context_lens.new_ones(size-args.batch)])
                bt=torch.cat([ctx.block_tables,ctx.block_tables.new_full((size-args.batch,ctx.block_tables.shape[1]),runner._graph_dummy_block)])
                set_context(False,slot_mapping=slots,context_lens=lens,block_tables=bt)
            current[0]=name; records[name]={}
            outputs[name]=runner.model.compute_logits(runner.model(ids,pos))[:args.batch].clone()
            current[0]=None
        runner.kv_cache.copy_(snapshot)
        outputs['graph_padded']=runner._replay_decode_graph(runner.decode_graphs[args.bucket],seqs).clone()
        live_mask=torch.ones(runner.kv_cache.shape[2]*runner.block_size,device='cuda',dtype=torch.bool)
        for seq in seqs: live_mask[seq.block_table[-1]*runner.block_size+seq.last_block_num_tokens-1]=False
        # Only the real last-token slots may change. No dummy writes allowed.
        before=snapshot.flatten(2,3);after=runner.kv_cache.flatten(2,3)
        kv_untouched=torch.equal(before[:,:,live_mask],after[:,:,live_mask])
        assert kv_untouched
        # Separate the local GEMM from RowParallelLinear's NCCL reduction.
        # Every rank repeats identical active rows at M=3 and M=4.
        x3 = records['eager_unpadded']['layer0.o_proj']['inputs'][0]
        weight = runner.model.model.layers[0].self_attn.o_proj.weight
        x4 = torch.cat([x3, x3.new_zeros((args.bucket-args.batch, x3.shape[1]))])
        local3 = torch.nn.functional.linear(x3, weight)
        local4 = torch.nn.functional.linear(x4, weight)
        reduced3, reduced4 = local3.clone(), local4.clone()
        dist.all_reduce(reduced3, group=groups['tp'])
        dist.all_reduce(reduced4, group=groups['tp'])
        fp3, fp4 = local3.float(), local4.float()
        dist.all_reduce(fp3, group=groups['tp'])
        dist.all_reduce(fp4, group=groups['tp'])
        isolated_collective = dict(rank=rank,
            local_gemm_max_abs=(local3.float()-local4[:args.batch].float()).abs().max().item(),
            bf16_reduce_max_abs=(reduced3.float()-reduced4[:args.batch].float()).abs().max().item(),
            fp32_reduce_max_abs=(fp3-fp4[:args.batch]).abs().max().item(),
            fp32_reduce_cast_equal=torch.equal(fp3.to(local3.dtype),fp4[:args.batch].to(local3.dtype)))
        Path(args.output).mkdir(parents=True,exist_ok=True)
        (Path(args.output)/f'collective_rank{rank}.json').write_text(json.dumps(isolated_collective,indent=2))
        if rank==0:
            def metrics(a,b):
                return dict(equal=torch.equal(a,b),max_abs=(a.float()-b.float()).abs().max().item(),
                            changed=int((a!=b).sum().item()),elements=a.numel())
            comparison={}
            for name in ('eager_padded','eager_padded_dummy_changed','eager_unpadded_repeat'):
                rows=[]
                for label,a in records['eager_unpadded'].items():
                    b=records[name][label]
                    ao,bo=a['output'],b['output']
                    if isinstance(ao,list): ao,bo=ao[0],bo[0]
                    row=dict(label=label,**metrics(ao,bo))
                    if '.expert_' in label:
                        row['input_metrics'] = metrics(a['inputs'], b['inputs'])
                    if label.endswith('.router'):
                        row['weight']=metrics(a['output'][1],b['output'][1])
                        row['same_expert_sets']=torch.equal(ao.sort(-1).values,bo.sort(-1).values)
                        if not row['equal']:
                            row['expert_ids']=ao.tolist()
                            row['padded_expert_ids']=bo.tolist()
                    rows.append(row)
                comparison[name]=rows
            result=dict(batch=args.batch, bucket=args.bucket, prefix_length=255,
                tp_reduction_dtype=os.environ['VLLM_LEARN_TP_REDUCTION_DTYPE'],
                layer_comparison=comparison,final={k:dict(**metrics(outputs['eager_unpadded'],v),
                argmax=v.argmax(-1).tolist()) for k,v in outputs.items()},
                padded_graph_vs_padded_eager=metrics(outputs['graph_padded'],outputs['eager_padded']),
                dummy_change=metrics(outputs['eager_padded'],outputs['eager_padded_dummy_changed']),kv_untouched=kv_untouched)
            # Replay the FIRST divergent linear in isolation, with exactly
            # the same active input rows and weights, and a FP32 oracle.
            first=next((r for r in comparison['eager_padded'] if not r['equal']),None)
            result['first_divergence']=first
            if first and first['label'].endswith(('q_proj','k_proj','v_proj','o_proj')):
                label=first['label'];layer_id=int(label.split('.')[0][5:]);proj=label.split('.')[1]
                module=getattr(runner.model.model.layers[layer_id].self_attn,proj)
                x3=records['eager_unpadded'][label]['inputs'][0]
                x4=torch.cat([x3,x3.new_zeros((args.bucket-args.batch, x3.shape[1]))])
                a=torch.nn.functional.linear(x3,module.weight)
                b=torch.nn.functional.linear(x4,module.weight)[:args.batch]
                af=torch.nn.functional.linear(x3.float(),module.weight.float())
                bf=torch.nn.functional.linear(x4.float(),module.weight.float())[:args.batch]
                result['isolated_linear']=dict(bf16=metrics(a,b),float32=metrics(af,bf),inputs_equal=torch.equal(x3,x4[:args.batch]))
            Path(args.output).mkdir(parents=True,exist_ok=True)
            (Path(args.output)/'padding_diagnosis.json').write_text(json.dumps(result,indent=2))
            print(json.dumps({k:v for k,v in result.items() if k!='layer_comparison'},indent=2),flush=True)
    for handle in handles:handle.remove()
    runner.shutdown();dist.destroy_process_group()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--port',type=int,default=31919)
    ap.add_argument('--batch',type=int,default=3);ap.add_argument('--bucket',type=int,default=4)
    ap.add_argument('--expert-intermediates', action='store_true')
    args=ap.parse_args()
    if not 0 < args.batch < args.bucket <= 8: ap.error('require 0 < batch < bucket <= 8')
    if os.getenv('PADDING_PROBE_FP32_TP') == '1':
        os.environ['VLLM_LEARN_TP_REDUCTION_DTYPE'] = 'float32'
    else:
        os.environ['VLLM_LEARN_TP_REDUCTION_DTYPE'] = 'model'
    os.environ.update(VLLM_LEARN_DECODE_GRAPH='1',VLLM_LEARN_EP_DENSE_DECODE='1',
        VLLM_LEARN_DENSE_EXPERT_BACKEND='bmm',VLLM_LEARN_REFERENCE_QK='0',
        VLLM_LEARN_ATTENTION_BACKEND='triton',VLLM_LEARN_EXPERT_BACKEND='torch_grouped',VLLM_LEARN_MM='0')
    import torch.multiprocessing as mp
    mp.spawn(worker,args=(args,),nprocs=8,join=True)


if __name__=='__main__':main()
