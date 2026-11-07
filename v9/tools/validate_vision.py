"""Real checkpoint vision/HF comparison plus complete-image chunk regression."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeVisionEncoderConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeVisionEncoder
    from layers.vision import Qwen3VisionTower
    from layers.mm_preprocess import image_to_patches
    from engine.model_runner import ModelRunner
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    ap.add_argument('--max-abs', type=float, default=0., help='maximum accepted component error')
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    raw = json.loads((Path(args.model) / 'config.json').read_text())
    cfg = raw['thinker_config']['vision_config']
    sd = {}
    for file in sorted(Path(args.model).glob('*.safetensors')):
        with safe_open(file, framework='pt', device='cpu') as f:
            for name in f.keys():
                if name.startswith('thinker.visual.'):
                    sd[name] = f.get_tensor(name)
    cfg['num_position_embeddings'] = sd['thinker.visual.pos_embed.weight'].shape[0]
    own = Qwen3VisionTower(cfg).eval()
    own.load_vision_state_dict(sd)
    hf_cfg = Qwen3OmniMoeVisionEncoderConfig(**cfg)
    hf_cfg._attn_implementation = 'sdpa'
    hf = Qwen3OmniMoeVisionEncoder(hf_cfg).eval()
    hsd = {}
    for name, value in sd.items():
        name = name.removeprefix('thinker.visual.')
        hsd[name] = value
    hf.load_state_dict(hsd, strict=True)
    del sd, hsd
    dtype = getattr(torch, args.dtype)
    # Module.to(BF16) also casts this nonpersistent frequency buffer. Preserve
    # its original FP32 values so this probe compares the intended rotary
    # math, rather than introducing an extra frequency-quantization policy.
    inv_freq = hf.rotary_pos_emb.inv_freq.clone().float()
    own = own.to('cuda', dtype); hf = hf.to('cuda', dtype)
    hf.rotary_pos_emb.inv_freq = inv_freq.cuda()
    rng = np.random.default_rng(712)
    rows, patches = [], []
    for height, width in ((96, 128), (128, 160)):
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        pixels, grid = image_to_patches(image)
        patches.append((pixels, grid))
        with torch.inference_mode():
            actual, deep = own(pixels.cuda(), grid.cuda())
            reference = hf(pixels.cuda(), grid.cuda())
        comparisons = []
        for name, a, b in [('merged', actual, reference.pooler_output), *[
            (f'deepstack{i}', a, b) for i, (a, b) in enumerate(zip(deep, reference.deepstack_features))]]:
            d = (a.float() - b.float()).abs()
            comparisons.append(dict(component=name, shape=list(a.shape), max_abs=d.max().item(),
                                    mean_abs=d.mean().item(), cosine=torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()))
            if d.max().item() > args.max_abs:
                raise AssertionError(comparisons[-1])
        rows.append(dict(height=height, width=width, grid=grid.tolist(), comparisons=comparisons))
    # Exercise production chunk assembly with this real encoder and two images.
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = 'cuda'; runner.config = SimpleNamespace(mm_feature_cache_bytes=256*1024**2)
    calls = []
    def encode(pixels, grid):
        calls.append(pixels.shape[0]); return own(pixels, grid)
    runner.model = SimpleNamespace(model=SimpleNamespace(vision_tower=own), run_vision=encode)
    runner._mrope_rows = lambda seqs: None
    images = []; pos = 2
    for p, g in patches:
        images.append(dict(pixels=p, grid=g[0].tolist(), pos=pos))
        pos += int(g.prod()) // 4 + 3
    seq = SimpleNamespace(seq_id=12, num_cached_tokens=0, num_scheduled_tokens=pos+2, mm={'images': images})
    expected = runner._build_mm_batch([seq]); calls.clear()
    pieces, dp, pp = [], [[], [], []], []
    for start in range(0, pos+2, 7):
        seq.num_cached_tokens = start; seq.num_scheduled_tokens = min(7, pos+2-start)
        got = runner._build_mm_batch([seq])
        if got is None: continue
        pieces.append(got[0]); pp.append(got[2]+start)
        for i, d in enumerate(got[1]): dp[i].append(d)
    torch.testing.assert_close(torch.cat(pieces), expected[0], rtol=0, atol=0)
    for i in range(3): torch.testing.assert_close(torch.cat(dp[i]), expected[1][i], rtol=0, atol=0)
    assert torch.equal(torch.cat(pp), expected[2]) and len(calls) == 2
    result = dict(model=args.model, torch=torch.__version__, dtype=args.dtype, rows=rows, chunk_exact=True,
                  chunk_encoder_calls=len(calls), cache_bytes_after=runner._vision_features.bytes,
                  scope='image encoder components and chunk assembly; not full-model HF parity')
    (out/'vision_validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
