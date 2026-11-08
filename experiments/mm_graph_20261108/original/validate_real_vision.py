"""Compare all real-image encoder components with the actual HF loader."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'v9'))
MODEL = Path('/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct')
MANIFEST = Path('/data/ifly_jyhe23/vllm-learn-graph-audit-20261108/datasets/chartqa/subset/manifest.json')
OUT = ROOT / 'outputs/completion_suite/vision_real_256'


def write(data):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / 'vision.json.tmp'
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(OUT / 'vision.json')


def main():
    import numpy as np
    import torch
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration
    from layers.vision import Qwen3VisionTower, vision_rope_cache
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(MODEL)
    # Use the same loader as the HF generation baseline. In particular, do
    # not cast the whole encoder with .to(BF16) or patch its rotary buffer.
    loaded = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={'': 'cuda:0'},
        attn_implementation='sdpa').eval()
    reference = loaded.visual
    del loaded
    torch.cuda.empty_cache()
    cfg = json.loads((MODEL / 'config.json').read_text())['thinker_config']['vision_config']
    weights = {}
    for file in sorted(MODEL.glob('*.safetensors')):
        with safe_open(file, framework='pt', device='cpu') as checkpoint:
            for name in checkpoint.keys():
                if name.startswith('thinker.visual.'):
                    weights[name] = checkpoint.get_tensor(name)
    cfg['num_position_embeddings'] = weights['thinker.visual.pos_embed.weight'].shape[0]
    cfg.setdefault('in_channels', 3)
    own = Qwen3VisionTower(cfg).eval()
    own.load_vision_state_dict(weights)
    del weights
    own = own.to('cuda:0', torch.bfloat16)
    inv = vision_rope_cache(own.dim // cfg['num_heads'] // 2, own.rope_base, 'cuda:0')
    frequency_exact = torch.equal(inv, reference.rotary_pos_emb.inv_freq)
    assert frequency_exact, 'actual HF rotary buffer differs'
    rows = json.loads(MANIFEST.read_text())['samples']
    result = dict(passed=False, torch=torch.__version__, dtype='bfloat16',
        samples=len(rows), frequency_exact=frequency_exact, rows=[],
        source_sha256=hashlib.sha256((ROOT/'v9/layers/vision.py').read_bytes()).hexdigest(),
        reference='unmodified full Thinker from_pretrained loader, sdpa; encoder extracted after loading',
        measurement='encoder components only; no full-model generation or throughput claim')
    begun = time.time()
    differences = []
    with torch.inference_mode():
        for max_pixels in (16384, 65536, 262144):
            for ordinal, row in enumerate(rows):
                with Image.open(MANIFEST.parent/row['image']) as image:
                    inputs = processor.image_processor(images=[image.convert('RGB')],
                        size={'shortest_edge':3136, 'longest_edge':max_pixels}, return_tensors='pt')
                pixels = inputs.pixel_values.cuda()
                grid = inputs.image_grid_thw
                actual, deep = own(pixels, grid)
                expected = reference(pixels, grid.cuda())
                components = []
                pairs = [('merged', actual, expected.pooler_output)] + [
                    (f'deepstack{i}', a, b) for i, (a, b) in enumerate(zip(deep, expected.deepstack_features))]
                assert len(deep) == len(expected.deepstack_features) == 3
                for name, a, b in pairs:
                    diff = (a.float()-b.float()).abs()
                    components.append(dict(component=name, shape=list(a.shape),
                        max_abs=diff.max().item(), mean_abs=diff.mean().item(), exact=torch.equal(a,b)))
                record = dict(index=row['index'], max_pixels=max_pixels,
                    grid=grid.tolist(), components=components)
                result['rows'].append(record)
                if not all(c['exact'] for c in components):
                    differences.append(record)
                if (ordinal+1) % 32 == 0:
                    write(result)
                    print('VISION',max_pixels,ordinal+1,'nonexact',len(differences),flush=True)
    result.update(passed=not differences, differences=differences, elapsed_s=time.time()-begun)
    write(result)
    assert not differences, f'{len(differences)} nonexact image/size comparisons'
    print('REAL_VISION_PASSED',len(result['rows']),flush=True)


if __name__ == '__main__':
    main()
