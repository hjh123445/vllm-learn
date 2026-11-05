"""Compare next-token logits on fixed HF continuations, avoiding drift.

Only diagnostic runs enable disk/CPU logits capture; not a speed benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--reference',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--ep',type=int,default=8)
    ap.add_argument('--pp',type=int,default=1)
    ap.add_argument('--batch',action='store_true')
    ap.add_argument('--gpu-memory-utilization',type=float,default=.70)
    ap.add_argument('--token-budget',type=int,default=512)
    ap.add_argument('--microbatch-tokens',type=int,default=128)
    ap.add_argument('--max-kv-blocks',type=int,default=None)
    args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    source=json.loads(Path(args.reference).read_text())
    os.environ['VLLM_LEARN_LOGITS_DIR']=str(out/'logits')
    from engine.core import EngineCore
    from samplingParams import SamplingParams
    core=EngineCore(args.model,ep_size=args.ep,pp_size=args.pp,master_port=31030,
                    pipeline_microbatch_tokens=args.microbatch_tokens,pipeline_lanes=2,
                    config_kwargs=dict(max_model_len=8192,max_num_batched_tokens=args.token_budget,
                    max_num_seqs=128,gpu_memory_utilization=args.gpu_memory_utilization,enforce_eager=True,
                    max_kvcache_blocks=args.max_kv_blocks,
                    enable_prefix_caching=False))
    records=[]
    cases=[(r,i) for r in source for i in range(len(r['output_ids']))]
    def submit(r,i):
        ids=r['input_ids']+r['output_ids'][:i]
        seq=core.add_request(ids,SamplingParams(temperature=0.,max_tokens=1,ignore_eos=True))
        records.append(dict(workload=r['workload'],index=r['index'],prefix_tokens=i,
                            seq_id=seq.seq_id,input_ids=ids,expected_next_token=r['output_ids'][i]))
    try:
        if args.batch:
            for r,i in cases:submit(r,i)
        completed={}
        for r,i in ([(None,None)] if args.batch else cases):
            if not args.batch:submit(r,i)
            while not core.is_finished():completed.update(dict(core.step()))
        for record in records:
            record['actual_next_token']=completed[record['seq_id']][0]
            record['greedy_matches_hf']=record['actual_next_token']==record['expected_next_token']
        (out/'requests.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
        (out/'config.json').write_text(json.dumps(dict(vars(args),
            ownership=os.getenv('VLLM_LEARN_EP_SHARD_REPLICATED_INPUT','0'),
            router_logits_dtype=os.getenv('VLLM_LEARN_ROUTER_LOGITS_DTYPE','float32'),
            reference_math=os.getenv('VLLM_LEARN_REFERENCE_MATH','0'),
            masked_gqa=os.getenv('VLLM_LEARN_SDPA_MASKED_GQA','0'),
            scope='fixed HF token prefixes; capture synchronizes; not performance data'),indent=2))
        print(json.dumps(dict(cases=len(records),greedy_matches=sum(r['greedy_matches_hf'] for r in records))))
    finally:core.shutdown()


if __name__=='__main__':main()
