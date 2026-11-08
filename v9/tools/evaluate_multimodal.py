"""Reproducible ChartQA quality and online/offline concurrency evaluation.

Quality runs stop at EOS; load runs use ignore_eos for exact output lengths.
Preprocessing, initialization, HTTP and engine timestamps have explicit scopes.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def quantiles(values):
    values = sorted(values)
    return {f'p{q}': values[round((len(values)-1)*q/100)] if values else None for q in (50, 95, 99)}


def score(prediction, labels):
    prediction = prediction.strip().lower().replace(',', '')
    prediction = re.sub(r'\s+', ' ', prediction)
    for label in labels:
        target = str(label).strip().lower().replace(',', '')
        if prediction == target:
            return True
        try:
            a, b = float(prediction.rstrip('%')), float(target.rstrip('%'))
            if (a == b) or (b != 0 and abs(a-b)/abs(b) <= .05):
                return True
        except ValueError:
            pass
    return False


def prepare(args):
    import pyarrow.parquet as pq
    table = pq.read_table(args.parquet)
    indices = random.Random(args.seed).sample(range(len(table)), args.samples)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in indices:
        row = table.slice(index, 1).to_pylist()[0]
        image = row['image']['bytes']
        name = f'image_{index:04d}.png'
        (out/name).write_bytes(image)
        rows.append(dict(index=index, image=name, image_sha256=hashlib.sha256(image).hexdigest(),
                         question=row['query'], answers=row['label'],
                         human_or_machine=row['human_or_machine']))
    write(out/'manifest.json', dict(dataset='HuggingFaceM4/ChartQA', split='test',
        seed=args.seed, source_sha256=hashlib.sha256(Path(args.parquet).read_bytes()).hexdigest(),
        total_split_rows=len(table), samples=rows))
    print('PREPARED', len(rows), out/'manifest.json', flush=True)


def messages(row, root, scope, context_tokens):
    text = row['question'] + '\nAnswer with the answer only, without explanation.'
    if context_tokens:
        text = ('Additional context: ' + 'This is a chart. ' * (context_tokens//5) + '\n') + text
    if scope == 'text':
        return [dict(role='user', content=text)]
    raw = (root/row['image']).read_bytes()
    return [dict(role='user', content=[dict(type='image_url', image_url=dict(
        url='data:image/png;base64,'+base64.b64encode(raw).decode())), dict(type='text', text=text)])]


def manifest(args):
    path = Path(args.manifest)
    data = json.loads(path.read_text(encoding='utf-8'))
    return data['samples'], path.parent, data


def own_inputs(args, row, root, tok, processor):
    from utils.chat_inputs import prepare_chat_inputs
    return prepare_chat_inputs(tok, messages(row, root, args.scope, args.context_tokens),
                               processor=processor, max_pixels=args.max_pixels)


def hf(args):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(args.model)
    rows, root, data = manifest(args)
    rows = rows[:args.samples]
    write(Path(args.output)/'provenance.json', provenance(args, data))
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval()
    results = []
    for row in rows:
        msg = messages(row, root, 'multimodal', 0)
        canonical = [{'role':'user','content':[{'type':'image'},
            {'type':'text','text':msg[0]['content'][1]['text']}]}]
        text = processor.apply_chat_template(canonical, tokenize=False, add_generation_prompt=True)
        with Image.open(root/row['image']) as image:
            inputs = processor(text=[text], images=[image.convert('RGB')],
                min_pixels=3136, max_pixels=args.max_pixels, return_tensors='pt')
        ids, mm = own_inputs(args, row, root, processor.tokenizer, processor)
        input_equal = ids == inputs.input_ids[0].tolist()
        pixel_ref = inputs['pixel_values'].float()
        pixel_own = torch.from_numpy(np.concatenate([im['pixels'] for im in mm['images']]))
        pixel_max_abs = (pixel_ref-pixel_own).abs().max().item()
        if pixel_max_abs != 0:
            raise AssertionError(dict(index=row['index'],pixel_max_abs=pixel_max_abs))
        grid_equal = inputs['image_grid_thw'].tolist() == [im['grid'] for im in mm['images']]
        pos3, delta = model.get_rope_index(input_ids=inputs.input_ids,
            image_grid_thw=inputs.image_grid_thw, attention_mask=inputs.attention_mask)
        positions_equal = torch.equal(pos3[:,0].cpu(), torch.from_numpy(mm['pos3']).long())
        delta_equal = int(delta.item()) == mm['delta']
        if not (input_equal and grid_equal and positions_equal and delta_equal):
            raise AssertionError(dict(index=row['index'],ids=input_equal,grids=grid_equal,
                                      positions=positions_equal,delta=delta_equal))
        inputs = {k: v.to('cuda:0') for k,v in inputs.items()}
        with torch.inference_mode():
            # The Omni root generation_config contains Talker settings only.
            # Give the standalone Thinker its tokenizer EOS explicitly.
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.output_tokens,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                return_dict_in_generate=bool(args.oracle_output), output_logits=bool(args.oracle_output))
        sequence = generated.sequences if args.oracle_output else generated
        output_ids = sequence[0, len(ids):].tolist()
        if args.oracle_output:
            with torch.inference_mode():
                vision = model.visual(inputs['pixel_values'], inputs['image_grid_thw'])
            torch.save(dict(input_ids=ids,mm=mm,output_ids=output_ids,
                step_logits=[t.detach().float().cpu() for t in generated.logits],
                merged=vision.pooler_output.float().cpu(),
                deepstack=[t.float().cpu() for t in vision.deepstack_features],
                vision_inv_freq=model.visual.rotary_pos_emb.inv_freq.float().cpu(),
                index=row['index'],torch=torch.__version__),args.oracle_output)
        answer = processor.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        result = dict(index=row['index'],input_ids=ids,output_ids=output_ids,answer=answer,
            correct=score(answer,row['answers']),labels=row['answers'],input_ids_equal=input_equal,
            pixel_max_abs=pixel_max_abs,grid_equal=grid_equal,mrope_equal=positions_equal,
            delta_equal=delta_equal)
        results.append(result)
        write(Path(args.output)/'quality.json',dict(backend='HF',model=args.model,
            torch=torch.__version__,samples=results,max_pixels=args.max_pixels,
            relaxed_accuracy=sum(r['correct'] for r in results)/len(results),
            frequency_policy='real from_pretrained loader; frequency buffers not patched'))
        print('HF',row['index'],repr(answer),result['correct'],'pixels',pixel_max_abs,flush=True)


def preset(args):
    os.environ.update(VLLM_LEARN_MM='1' if args.scope == 'multimodal' else '0',
        VLLM_LEARN_BATCH_INVARIANT='1', VLLM_LEARN_REFERENCE_QK='0',
        VLLM_LEARN_ATTENTION_BACKEND='triton', VLLM_LEARN_EP_DENSE_DECODE='0', VLLM_LEARN_EP_SPARSE_DECODE='1',
        VLLM_LEARN_DENSE_EXPERT_BACKEND='bmm',VLLM_LEARN_EXPERT_BACKEND='torch_grouped',
        VLLM_LEARN_DECODE_GRAPH='1' if args.graph else '0')


def provenance(args, data):
    return dict(options=vars(args),dataset={k:v for k,v in data.items() if k!='samples'},
        numerical_policy={k:v for k,v in os.environ.items() if k.startswith('VLLM_LEARN_')},
        source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).resolve().parents[1].rglob('*.py')
            if '__pycache__' not in p.parts})


def offline(args):
    preset(args)
    from transformers import AutoTokenizer, AutoProcessor
    from engine.core import EngineCore
    from samplingParams import SamplingParams
    from utils.observed_runner import ObservedModelRunner
    from utils.observation_config import configure_observation
    rows, root, data = manifest(args)
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    configure_observation(out, profile=False)
    write(out/'provenance.json',provenance(args,data))
    tok = AutoTokenizer.from_pretrained(args.model)
    template = Path(args.model)/'chat_template.json'
    if template.exists():
        tok.chat_template = json.loads(template.read_text())['chat_template']
    processor = AutoProcessor.from_pretrained(args.model) if args.scope == 'multimodal' else None
    initialized = time.perf_counter()
    core = EngineCore(args.model,tp_size=8,enable_expert_parallel=True,master_port=args.master_port,
        runner_cls=ObservedModelRunner,config_kwargs=dict(max_num_seqs=args.concurrency,
            max_num_batched_tokens=args.prefill_tokens,max_model_len=args.max_model_len,
            max_kvcache_blocks=args.kv_blocks,gpu_memory_utilization=.75,
            decode_graph_reserve_bytes=2*2**30,enable_prefix_caching=False))
    initialization_s = time.perf_counter()-initialized
    results = []
    selected = rows[:args.samples] if args.quality else [rows[i % len(rows)] for i in range(args.requests)]
    try:
        # Warm up with a real complete image, without retaining its request cache.
        ids, mm = own_inputs(args, selected[0],root,tok,processor)
        core.add_request(ids,SamplingParams(temperature=0.,max_tokens=2,ignore_eos=True),mm=mm)
        while not core.is_finished():core.step()
        for repeat in range(args.repeats):
            begun=time.perf_counter(); prepared=[]
            for request_index, row in enumerate(selected):
                started=time.perf_counter()
                ids,mm=own_inputs(args,row,root,tok,processor)
                prepared.append((request_index,row,ids,mm,started,time.perf_counter()))
            prepared_end=time.perf_counter()
            # Closed-loop concurrency: admit at most C live requests, refill
            # slots after completions. A 2*C run is more than a single burst.
            pending=list(prepared);active={};times={};records=[];peak_active=0
            engine_start=time.perf_counter()
            while pending or active:
                while pending and len(active)<args.concurrency:
                    request_index,row,ids,mm,started,processed=pending.pop(0)
                    sp=SamplingParams(temperature=0.,max_tokens=args.output_tokens,ignore_eos=not args.quality)
                    seq=core.add_request(ids,sp,mm=mm)
                    active[seq.seq_id]=(request_index,row,seq,started,processed,time.perf_counter())
                    times[seq.seq_id]=[]
                peak_active=max(peak_active,len(active))
                completed=dict(core.step())
                for event in core.last_step_events:
                    if event['valid_decode_token'] and event['seq_id'] in times:
                        times[event['seq_id']].append(event['step_end'])
                for sid,output_ids in completed.items():
                    request_index,row,seq,started,processed,admitted=active.pop(sid)
                    token_times=times.pop(sid)
                    if len(token_times) != len(output_ids) or (not args.quality and len(output_ids) != args.output_tokens):
                        raise AssertionError('output length or engine token timestamps disagree')
                    answer=tok.decode(output_ids,skip_special_tokens=True).strip()
                    records.append(dict(request_index=request_index,index=row['index'],output_ids=output_ids,answer=answer,
                        correct=score(answer,row['answers']) if args.quality else None,
                        prompt_tokens=seq.num_prompt_tokens,output_tokens=len(output_ids),
                        preprocessing_s=processed-started,ttft_s=token_times[0]-admitted,
                        e2e_s=token_times[-1]-admitted,arrival_s=[t-admitted for t in token_times]))
                    records[-1].update(pipeline_ttft_s=token_times[0]-begun,
                                       pipeline_e2e_s=token_times[-1]-begun)
            end=time.perf_counter()
            record=summary(records,end-engine_start,args)
            record.update(repeat=repeat,initialization_s=initialization_s,peak_active=peak_active,
                preprocessing_wall_s=prepared_end-begun,pipeline_wall_s=end-begun,
                pipeline_output_tok_s=sum(r['output_tokens'] for r in records)/(end-begun),
                pipeline_ttft_s=quantiles([r['pipeline_ttft_s'] for r in records]),
                pipeline_e2e_s=quantiles([r['pipeline_e2e_s'] for r in records]),
                measurement=('engine window includes vision + Thinker' if args.scope == 'multimodal'
                    else 'engine window includes text-only Thinker') + '; pipeline also includes input preprocessing',
                quality_accuracy=sum(r['correct'] for r in records)/len(records) if args.quality else None,
                requests=records)
            results.append(record);write(out/'results.json',results)
            print(json.dumps({k:v for k,v in record.items() if k!='requests'}),flush=True)
    finally:core.shutdown()


def summary(records,elapsed,args):
    good=[r for r in records if 'error' not in r]
    tokens=sum(r['output_tokens'] for r in good)
    itl=[b-a for r in good for a,b in zip(r['arrival_s'],r['arrival_s'][1:])]
    return dict(scope=args.scope,backend=args.mode,concurrency=args.concurrency,max_pixels=args.max_pixels,
        context_tokens=args.context_tokens,target_output_tokens=args.output_tokens,graph=args.graph,
        wall_s=elapsed,completed=len(good),errors=len(records)-len(good),output_tokens=tokens,
        output_tok_s=tokens/elapsed,request_s=len(good)/elapsed,
        ttft_s=quantiles([r['ttft_s'] for r in good]),e2e_s=quantiles([r['e2e_s'] for r in good]),
        itl_s=quantiles(itl))


def online(args):
    rows,root,data=manifest(args);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    write(out/'provenance.json',provenance(args,data))
    selected=[rows[i % len(rows)] for i in range(args.requests)]
    def request(item):
        request_index, row = item
        started=time.perf_counter()
        try:
            body=json.dumps(dict(model=args.model,messages=messages(row,root,args.scope,args.context_tokens),
                max_tokens=args.output_tokens,temperature=0.,ignore_eos=True,stream=True,
                mm_processor_kwargs=dict(max_pixels=args.max_pixels),
                stream_options=dict(include_usage=True))).encode()
            req=urllib.request.Request(args.url.rstrip('/')+'/v1/chat/completions',body,
                headers={'Content-Type':'application/json'})
            arrivals=[];parts=[];usage=None;done=False;request_id=None
            with urllib.request.urlopen(req,timeout=args.timeout) as response:
                for line in response:
                    if not line.startswith(b'data: '):continue
                    raw=line[6:].strip()
                    if raw==b'[DONE]':done=True;break
                    chunk=json.loads(raw)
                    if 'error' in chunk:raise RuntimeError(chunk['error'])
                    if request_id is None:request_id=chunk['id']
                    elif request_id != chunk['id']:raise RuntimeError('SSE request id changed')
                    if chunk.get('usage'):usage=chunk['usage']
                    for choice in chunk['choices']:
                        content=choice['delta'].get('content','')
                        if content:arrivals.append(time.perf_counter()-started);parts.append(content)
            ended=time.perf_counter()
            if usage is None or not arrivals or not done:raise RuntimeError('missing terminal usage, DONE or output content')
            if usage['completion_tokens'] != args.output_tokens:raise RuntimeError('unexpected output length')
            return dict(request_index=request_index,request_id=request_id,index=row['index'],answer=''.join(parts),output_tokens=usage['completion_tokens'],
                prompt_tokens=usage['prompt_tokens'],ttft_s=arrivals[0],e2e_s=ended-started,arrival_s=arrivals)
        except Exception as exc:return dict(request_index=request_index,index=row['index'],error=repr(exc),e2e_s=time.perf_counter()-started)
    results=[]
    for repeat in range(args.repeats):
        started=time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool: records=list(pool.map(request,enumerate(selected)))
        result=summary(records,time.perf_counter()-started,args)
        result.update(repeat=repeat,measurement='closed-loop HTTP client including image read/base64/upload, server preprocessing, queue, vision, Thinker, streaming; TTFT/ITL measure visible content chunks',requests=records)
        results.append(result);write(out/'results.json',results)
        print(json.dumps({k:v for k,v in result.items() if k!='requests'}),flush=True)
    if any(r['errors'] for r in results):raise RuntimeError('request errors recorded in results.json')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('prepare','hf','offline','online'),required=True)
    ap.add_argument('--model')
    ap.add_argument('--output',required=True)
    ap.add_argument('--manifest')
    ap.add_argument('--parquet')
    ap.add_argument('--samples',type=int,default=256)
    ap.add_argument('--seed',type=int,default=20261108)
    ap.add_argument('--scope',choices=('text','multimodal'),default='multimodal')
    ap.add_argument('--quality',action='store_true')
    ap.add_argument('--graph',action='store_true')
    ap.add_argument('--concurrency',type=int,default=128)
    ap.add_argument('--requests',type=int,default=256)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--max-pixels',type=int,default=262144)
    ap.add_argument('--context-tokens',type=int,default=0)
    ap.add_argument('--output-tokens',type=int,default=64)
    ap.add_argument('--prefill-tokens',type=int,default=2048)
    ap.add_argument('--max-model-len',type=int,default=4096)
    ap.add_argument('--kv-blocks',type=int,default=2048)
    ap.add_argument('--master-port',type=int,default=31935)
    ap.add_argument('--url',default='http://127.0.0.1:8018')
    ap.add_argument('--timeout',type=float,default=1200.)
    ap.add_argument('--oracle-output',help='HF only: save generation step logits and actual loader vision features')
    args=ap.parse_args()
    if min(args.samples,args.concurrency,args.requests,args.repeats,args.output_tokens)<1:ap.error('counts must be positive')
    if args.mode!='prepare' and (not args.model or not args.manifest):ap.error('model and manifest required')
    if args.oracle_output and (args.mode != 'hf' or args.samples != 1):ap.error('oracle-output requires HF with samples=1')
    globals()[args.mode](args)


if __name__=='__main__':main()
