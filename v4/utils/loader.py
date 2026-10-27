# loader.py
import os
import glob

import torch
import torch.nn as nn
from safetensors import safe_open


def default_weight_loader(param, loaded_weight):
    assert param.shape == loaded_weight.shape, (param.shape, loaded_weight.shape)
    param.data.copy_(loaded_weight)


def shard_weight_loader(param, loaded_weight, shard_id):
    """融合矩阵按行写入：shard 0 (gate) 写上半，shard 1 (up) 写下半"""
    size = loaded_weight.shape[0]                       # intermediate_size
    offset = 0 if shard_id == 0 else size
    param.data.narrow(0, offset, size).copy_(loaded_weight)


# checkpoint名 → (模型名, 分片号) 的映射表；名字相同的参数走直装
PACKED_MAPPING = {
    "gate_proj": ("fc1", 0),
    "up_proj":   ("fc1", 1),
}

# checkpoint名 → 模型名 的普通改名
RENAME_MAPPING = {
    "down_proj": "fc2",
}


def map_name(ckpt_name: str):
    for k, (target, shard) in PACKED_MAPPING.items():
        if k in ckpt_name:
            return ckpt_name.replace(k, target), shard
    for k, target in RENAME_MAPPING.items():
        if k in ckpt_name:
            return ckpt_name.replace(k, target), None
    return ckname_fix(ckpt_name), None


def ckname_fix(name: str) -> str:
    # 如果你的模型属性名和 HF 完全一致（self_attn/mlp/...），这里原样返回
    return name


def load_model(model: nn.Module, model_dir: str,
               device="cuda", dtype=torch.bfloat16,
               tie_word_embeddings: bool = True):
    """CPU 上装载 + 对账 → 通过后再迁移到 GPU。失败早退出。"""
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    assert files, f"没找到 safetensors 文件: {model_dir}"

    loaded = set()                                     # 记账本
    for file in files:                                 # 天然支持分片 checkpoint
        with safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                param_name, shard_id = map_name(name)
                param = model.get_parameter(param_name)
                w = f.get_tensor(name)
                if shard_id is None:
                    default_weight_loader(param, w)
                else:
                    shard_weight_loader(param, w, shard_id)
                loaded.add(param_name)

    # ============ 对账：唯一能抓住"漏装"的保险 ============
    model_keys = set(model.state_dict().keys())
    allowed_missing = {"lm_head.weight"} if tie_word_embeddings else set()

    not_loaded = model_keys - loaded - allowed_missing
    assert not not_loaded, f"以下参数从未被加载（还是 torch.empty 的垃圾值）:\n{not_loaded}"

    # 抽查一个，确认 copy_ 真的生效了
    with safe_open(files[0], "pt", "cpu") as f:
        k0 = next(iter(f.keys()))
        assert torch.equal(model.get_parameter(map_name(k0)[0]).cpu().float(),
                           f.get_tensor(k0).float()), "权重内容校验失败"

    # tie embeddings：lm_head 和 embed_tokens 共享同一份权重
    if tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model