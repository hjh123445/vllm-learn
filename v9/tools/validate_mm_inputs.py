"""Compare real image/chat/MRoPE inputs against installed HF, without weights."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration
    from tools.evaluate_multimodal import own_inputs, write
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--samples',type=int,default=256)
    ap.add_argument('--max-pixels',type=int,nargs='+',default=[16384,65536,262144])
    args=ap.parse_args()
    torch.set_num_threads(2)
    processor=AutoProcessor.from_pretrained(args.model)
    cfg=AutoConfig.from_pretrained(args.model).thinker_config
    # get_rope_index is host-only geometry; call the real HF method with its
    # config rather than copying its implementation into a test oracle.
    oracle=Qwen3OmniMoeThinkerForConditionalGeneration.__new__(Qwen3OmniMoeThinkerForConditionalGeneration)
    torch.nn.Module.__init__(oracle)
    oracle.config=cfg
    oracle.spatial_merge_size=cfg.vision_config.spatial_merge_size
    root=Path(args.manifest).parent
    rows=json.loads(Path(args.manifest).read_text())['samples'][:args.samples]
    results=[]
    for max_pixels in args.max_pixels:
        options=SimpleNamespace(scope='multimodal',context_tokens=0,max_pixels=max_pixels)
        for row in rows:
            ids,mm=own_inputs(options,row,root,processor.tokenizer,processor)
            text=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},
                {'type':'text','text':row['question']+'\nAnswer with the answer only, without explanation.'}]}],
                tokenize=False,add_generation_prompt=True)
            with Image.open(root/row['image']) as image:
                inputs=processor(text=[text],images=[image.convert('RGB')],
                    images_kwargs={'size':{'shortest_edge':3136,'longest_edge':max_pixels}},return_tensors='pt')
            pixel=torch.from_numpy(mm['images'][0]['pixels'])
            pos3,delta=Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index(oracle,
                input_ids=inputs.input_ids,image_grid_thw=inputs.image_grid_thw,
                attention_mask=inputs.attention_mask)
            record=dict(index=row['index'],max_pixels=max_pixels,prompt_tokens=len(ids),
                grid=mm['images'][0]['grid'],ids_equal=ids==inputs.input_ids[0].tolist(),
                grid_equal=inputs.image_grid_thw.tolist()==[im['grid'] for im in mm['images']],
                pixels_equal=torch.equal(pixel,inputs.pixel_values),
                max_abs=(pixel-inputs.pixel_values).abs().max().item(),
                positions_equal=torch.equal(pos3[:,0].long(),torch.from_numpy(mm['pos3']).long()),
                delta_equal=int(delta.item())==mm['delta'])
            results.append(record)
            if not all(record[k] for k in ('ids_equal','grid_equal','pixels_equal','positions_equal','delta_equal')):
                write(args.output,dict(passed=False,rows=results));raise AssertionError(record)
        print('INPUTS_PASSED',max_pixels,len(rows),flush=True)
    write(args.output,dict(passed=True,rows=results,samples=len(rows),
        measurement='real HF processor and get_rope_index; no model weights or generation'))


if __name__=='__main__':main()
