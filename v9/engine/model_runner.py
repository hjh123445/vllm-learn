# v5/engine/model_runner.py
"""v3 ModelRunner（重写版，单卡单进程，无 mp/分布式）。

职责与 nano 的 model_runner 相同：载模型 → warmup 量激活峰值 →
warmup 实测激活峰值 → 按显存预算公式算 num_kvcache_blocks → 分配池子并注入每层 attention →
每步 prepare（构造 Context）→ 前向 → 采样。
区别：模型是 v2 的实现（平铺 + 纯 PyTorch attention），采样支持逐请求温度（含 greedy）。
"""
import os

import torch

from config import Config
from engine.sequence import Sequence
from models.qwen3_moe import Qwen3MoeForCausalLM
from utils.context import set_context, get_context, reset_context
from utils.loader import load_model, load_thinker_config
from engine.sampler import Sampler


def _is_dsv4(model_path: str, hf) -> bool:
    """DSV4 判定（P0-1 修复：本条件引用 os —— 文件此前没有 import os，
    Qwen 路径求值 os.path.basename 直接 NameError）。"""
    return getattr(hf, "model_type", "") == "deepseek_v4" or \
        "DeepSeek-V4" in os.path.basename(model_path)


def _load_hf_config(config: Config):
    """hf_config 归一化（P0-1 修复）：旧代码无条件 load_thinker_config()，
    但普通 Qwen3 / DSV4 的 config.json 没有 thinker_config——在进入模型
    分派之前就 KeyError。现在：Omni（有 thinker_config.text_config）平铺
    text_config；其余模型用 AutoConfig 属性 + config.json 平铺。"""
    from types import SimpleNamespace
    import json
    with open(os.path.join(config.model, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw.get("thinker_config"), dict) and \
            isinstance(raw["thinker_config"].get("text_config"), dict):
        hf = SimpleNamespace(**load_thinker_config(config.model))
    else:
        attrs = {k: v for k, v in config.hf_config.__dict__.items()}
        attrs.update({k: v for k, v in raw.items()})
        hf = SimpleNamespace(**attrs)
    hf.dtype = getattr(hf, "dtype", None) or torch.bfloat16
    return hf


class ModelRunner:

    def __init__(self, config: Config, dtype=None):
        self.config = config
        # hf_config 归一化：Omni 平铺 thinker text_config；普通模型平铺 config.json
        from types import SimpleNamespace
        hf = _load_hf_config(config)
        config.hf_config = hf
        self.block_size = config.kvcache_block_size
        self.device = "cuda"
        self.dtype = dtype or hf.dtype

        # ---- PP 视角属性（收敛 4 处 getattr(STATE,"pp_size") 重复）----
        # launcher 在构造 ModelRunner 之前 configure_parallel()——这里读到的
        # 即本 rank 的最终坐标。单卡退化：pp=1 时 _pp_layers=(0, n)，
        # _is_pp_last_stage=True，后续判断零分支。
        from layers.parallel import STATE, get_pp_indices
        self._n_layers_total = hf.num_hidden_layers
        if getattr(STATE, "pp_size", 1) > 1:
            self._pp_layers = get_pp_indices(self._n_layers_total,
                                             STATE.pp_rank, STATE.pp_size)
        else:
            self._pp_layers = (0, self._n_layers_total)
        self._n_layers_local = self._pp_layers[1] - self._pp_layers[0]
        self._is_pp_last_stage = STATE.pp_rank == STATE.pp_size - 1

        # ---- 载模型（MoE：CPU 构造 + 对账装载（专家融合 w13/w2）+ 迁移）----
        # 模型选择（多模型共存不互斥）: DSV4-Flash 走专属 loader（MXFP4/FP8
        # 双格式反量化 + 分层分放 layer_device_map），其余走 Qwen thinker 路径
        if _is_dsv4(config.model, hf):
            from models.deepseek_v4 import DSV4ForCausalLM
            from utils.dsv4_loader import load_dsv4
            self.model = DSV4ForCausalLM(config.model)
            load_dsv4(self.model, config.model, device=self.device,
                      dtype=self.dtype)
        else:
            self.model = Qwen3MoeForCausalLM(hf, config.max_model_len)
            load_model(self.model, config.model, device=self.device, dtype=self.dtype)
        self.model.eval()
        self.sampler = Sampler()

        # ---- 可选：装载后、分池前在线量化（W4）----
        # 顺序关键（WORKLOG §3.3 缺口的根修）：量化省出的显存必须在 allocate
        # 之前生效——否则池子按 bf16 预算分配，省余量不回流（quantize 在
        # EngineCore 构造后调用的旧路径就是这个问题）。
        if getattr(config, "quantize_int4_w4", False):
            n = self.model.quantize_int4(group=128)
            torch.cuda.empty_cache()
            print(f"[runner] W4 在线量化 {n} 层（分池前，省余量回流 KV 预算）",
                  flush=True)

        # ---- warmup 量激活峰值 → 按显存预算公式定池子 ----
        # v9.3 组件化并行：模型构造时就只有本 rank 的 1/N 权重（不再有
        # "全量加载再切分"的阶段）——warmup 可以在任何配置下直接跑。
        # launcher 责任：构造 ModelRunner 之前 configure_parallel()
        self.kv_cache = None
        self._warmup_model()
        self.allocate_kv_cache()

    # ================= 显存 =================

    def _assign_layer_caches(self, kv_cache):
        """把池子的每层切片指给对应 attention（k_cache [blocks, bs, kvh, hd]）。
        PP：只数本 stage 的真层（PPMissingLayer 无 cache 属性）——池子按本
        stage 层数分配，显存不按全量 48 层虚算。"""
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1
        assert layer_id == self._n_layers_local, \
            f"{layer_id} != 本 stage 层数 {self._n_layers_local}"

    def _per_rank_kv_heads(self):
        """per-rank kv 头数：从模型第一层 attention 组件读（v9.3 组件构造时
        已按 ParallelState 切——TP2 下 kv=4 → 每 rank 2；kv 复制时 ≥1）。
        取代 v6 的 _shard_kv_heads_for_tp 事后补丁（引擎读组件，不是组件等引擎）。"""
        for m in self.model.modules():
            if hasattr(m, "num_kv_head"):
                return m.num_kv_head
        return self.config.hf_config.num_key_value_heads

    def _warmup_model(self):
        """dummy 最大压力前向量激活峰值（prefill + decode 两种形状都量到）。
        用 64 块临时池 + 手动指好的 block 表——我们的 attention 要从 cache 读，
        不像 nano 的 flash prefill 可以无 cache 直算。"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        hf = self.config.hf_config
        bs = self.block_size
        seq_len = min(self.config.max_num_batched_tokens, self.config.max_model_len)
        num_seqs = min(self.config.max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        kvh = self._per_rank_kv_heads()
        # PP：本 stage 层数（池子/块字节数按 stage 层算）
        n_layers = self._n_layers_local
        hd = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        blocks_per_seq = (seq_len + bs - 1) // bs
        tmp = torch.empty(2, n_layers, num_seqs * blocks_per_seq, bs, kvh, hd,
                          dtype=self.dtype, device=self.device)
        self._assign_layer_caches(tmp)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for i, seq in enumerate(seqs):
            seq.block_table = [i * blocks_per_seq + j for j in range(blocks_per_seq)]
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        for seq in seqs:                              # decode 形状
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
        self.run(seqs, False)
        # 记录激活峰值：此时 peak/current 都含 tmp，差值恰好隔离出 A（数学上 T 抵消）
        stats = torch.cuda.memory_stats()
        self._activation_peak = stats["allocated_bytes.all.peak"] - stats["allocated_bytes.all.current"]
        # 释放临时池：_assign_layer_caches 把 tmp 的 view 挂在模块上，函数返回
        # 后 tmp 靠这些 view 存活——必须显式打破引用才真正归还显存。
        # （不释放的话 allocate 里量的 used 会把 T 当成常驻显存多扣一遍，
        #   池子恰好小 T/block_bytes 个块——v9.1 实测少 64 blocks / 17%）
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = torch.tensor([], device=self.device)
                module.v_cache = torch.tensor([], device=self.device)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """池子预算 = util×总显存 − 常驻(权重+context+其他进程) − warmup 实测激活峰值。

        v9 组件化后唯一路径（组件构造即分片，warmup 在任何 TP/EP 配置下都能
        直接跑，_activation_peak 必然已设置）——v6 的"TP 手动路径（无 warmup，
        6GB 启发式）"与"重分配（陈旧 peak）"两个分支已删：它们服务的手动
        allocate 入口（run_tp/切分后重分池）在并行架构清理时一并移除。"""
        config = self.config
        hf = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        num_kv_heads = self._per_rank_kv_heads()
        head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        _n = self._n_layers_local
        block_bytes = 2 * _n * self.block_size * num_kv_heads * head_dim * self.dtype.itemsize
        # tmp 已在 warmup 末尾释放：used 只含 权重+context+其他进程，
        # A 用实测峰值——预算 = util×total − 常驻 − 激活峰值
        budget = int(total * config.gpu_memory_utilization) - used - self._activation_peak
        config.num_kvcache_blocks = budget // block_bytes
        if config.num_kvcache_blocks <= 0:
            act = self._activation_peak / 1e9
            print(f"[池诊断] total={total/1e9:.1f} used={used/1e9:.1f} "
                  f"activation_peak={act:.1f} block={block_bytes/1e6:.1f}MB "
                  f"budget={budget/1e9:.1f}GB", flush=True)
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, _n, config.num_kvcache_blocks,
                                    self.block_size, num_kv_heads, head_dim,
                                    dtype=self.dtype, device=self.device)
        self._assign_layer_caches(self.kv_cache)
        print(f"[v3 显存预算] 总显存 {total/1e9:.1f} GB × 利用率 {config.gpu_memory_utilization}")
        print(f"  → KV 池 {config.num_kvcache_blocks} blocks × {self.block_size} tok "
              f"({self.kv_cache.numel() * self.dtype.itemsize / 1e9:.2f} GB, "
              f"每 block {block_bytes/1e6:.2f} MB, "
              f"激活预算 实测 {self._activation_peak/1e9:.2f} GB")

    # ================= 每步的 prepare（构造 Context）=================

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables, dtype=torch.int32).to(self.device)

    def prepare_prefill(self, seqs: list[Sequence]):
        """平铺 prefill 批（支持 chunked prefill / 前缀缓存：q 是 k 的尾段）。"""
        input_ids, positions = [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_seqlen_q = max_seqlen_k = 0
        slot_mapping = []
        bs = self.block_size
        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seq.num_scheduled_tokens)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(max_seqlen_q, seq.num_scheduled_tokens)
            max_seqlen_k = max(max_seqlen_k, end)
            for pos in range(start, end):
                slot_mapping.append(seq.block_table[pos // bs] * bs + pos % bs)
        block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64).to(self.device)
        positions = torch.tensor(positions, dtype=torch.int64).to(self.device)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32).to(self.device)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32).to(self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).to(self.device)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids, positions, slot_mapping, context_lens = [], [], [], []
        bs = self.block_size
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * bs + seq.last_block_num_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64).to(self.device)
        positions = torch.tensor(positions, dtype=torch.int64).to(self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).to(self.device)
        context_lens = torch.tensor(context_lens, dtype=torch.int32).to(self.device)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens,
                    block_tables=block_tables)
        return input_ids, positions

    # ================= 前向 + 采样 =================

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
        hidden = self.model(input_ids, positions)                 # [total_q, hidden]
        # PP 中间 stage：hidden 已发给下游，跳过 lm_head/采样（worker 只在
        # last stage 发 response）
        if not self._is_pp_last_stage:
            return [0] * len(seqs)
        if is_prefill:
            last = get_context().cu_seqlens_q[1:] - 1              # 每序列最后一个 q 位置
            logits = self.model.compute_logits(hidden[last])
        else:
            logits = self.model.compute_logits(hidden)
        temperatures = torch.tensor([seq.temperature for seq in seqs],
                                    dtype=torch.float32, device=self.device)
        return self.sampler(logits, temperatures).tolist()

