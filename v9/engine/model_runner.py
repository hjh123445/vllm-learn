# v5/engine/model_runner.py
"""v3 ModelRunner（重写版，单卡单进程，无 mp/分布式）。

职责与 nano 的 model_runner 相同：载模型 → warmup 量激活峰值 →
按 nano 的公式算 num_kvcache_blocks → 分配池子并注入每层 attention →
每步 prepare（构造 Context）→ 前向 → 采样。
区别：模型是 v2 的实现（平铺 + 纯 PyTorch attention），采样支持逐请求温度（含 greedy）。
"""
import torch

from config import Config
from engine.sequence import Sequence
from models.qwen3_moe import Qwen3MoeForCausalLM
from utils.context import set_context, get_context, reset_context
from utils.loader import load_model, load_thinker_config
from engine.sampler import Sampler


class ModelRunner:

    def __init__(self, config: Config, dtype=None):
        self.config = config
        # thinker 的 text_config 是嵌套 dict → 平铺成 SimpleNamespace（模型文件按属性读）
        from types import SimpleNamespace
        t = load_thinker_config(config.model)
        hf = SimpleNamespace(**t)
        hf.dtype = getattr(hf, "dtype", None) or torch.bfloat16
        config.hf_config = hf
        self.block_size = config.kvcache_block_size
        self.device = "cuda"
        self.dtype = dtype or hf.dtype

        # ---- 载模型（MoE：CPU 构造 + 对账装载（专家融合 w13/w2）+ 迁移）----
        self.model = Qwen3MoeForCausalLM(hf, config.max_model_len)
        load_model(self.model, config.model, device=self.device, dtype=self.dtype)
        self.model.eval()
        self.sampler = Sampler()

        # ---- warmup 量激活峰值 → 按 nano 的公式定池子 ----
        # TP 模式下 warmup 在切分前跑会按全量权重分配爆显存 → 跳过 warmup，
        # 池子由 run_tp.py 在切分后手动分配
        self.kv_cache = None
        if getattr(config, "tensor_parallel_size", 1) > 1 or \
           getattr(config, "expert_parallel_size", 1) > 1:
            return          # TP/EP: 切分后由启动器手动分配
        self._warmup_model()
        self.allocate_kv_cache()

    # ================= 显存 =================

    def _assign_layer_caches(self, kv_cache):
        """把池子的每层切片指给对应 attention（k_cache [blocks, bs, kvh, hd]）。"""
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1
        assert layer_id == self.config.hf_config.num_hidden_layers

    def _shard_kv_heads_for_tp(self, tp_rank: int, tp_size: int):
        """TP 切分后 attention 的 kv 头数变 kv/TP——池子和 Attention 模块同步缩。
        在 shard_model_for_tp 之后、allocate_kv_cache 之前调用。"""
        hf = self.config.hf_config
        # kv 头数 < TP 时复制（parallel.py 同语义）：per_kv = max(1, kv/TP)
        per_kv = max(1, hf.num_key_value_heads // tp_size)
        for m in self.model.modules():
            if hasattr(m, "num_kv_head"):
                m.num_kv_head = per_kv
        self._tp_kv_heads = per_kv

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
        kvh = hf.num_key_value_heads
        hd = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        blocks_per_seq = (seq_len + bs - 1) // bs
        tmp = torch.empty(2, hf.num_hidden_layers, num_seqs * blocks_per_seq, bs, kvh, hd,
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
        """池子预算。单卡路径 = nano 公式（warmup 量激活峰值）；
        TP 手动路径（无 warmup）= util×total − 其他进程 − 本进程权重 − 激活余量(6GB)。"""
        config = self.config
        hf = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = getattr(self, "_tp_kv_heads", None) or hf.num_key_value_heads
        head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        block_bytes = 2 * hf.num_hidden_layers * self.block_size * num_kv_heads * head_dim * self.dtype.itemsize
        if getattr(self, "_activation_peak", None) is not None:
            # warmup 路径（tmp 已在 warmup 末尾释放）：used 只含 权重+context+其他进程，
            # A 用实测峰值——预算 = util×total − 常驻 − 激活峰值
            budget = int(total * config.gpu_memory_utilization) - used - self._activation_peak
        elif self.kv_cache is None:
            # TP 手动路径（无 warmup）：激活余量用 6GB 启发式
            budget = int(total * config.gpu_memory_utilization) - (used - current) - current - 6 * 10**9
        else:
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            budget = int(total * config.gpu_memory_utilization) - used - peak + current
        config.num_kvcache_blocks = budget // block_bytes
        if config.num_kvcache_blocks <= 0:
            print(f"[池诊断] total={total/1e9:.1f} used={used/1e9:.1f} current={current/1e9:.1f} "
                  f"block={block_bytes/1e6:.1f}MB budget={budget/1e9:.1f}GB", flush=True)
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf.num_hidden_layers, config.num_kvcache_blocks,
                                    self.block_size, num_kv_heads, head_dim,
                                    dtype=self.dtype, device=self.device)
        self._assign_layer_caches(self.kv_cache)
        print(f"[v3 显存预算] 总显存 {total/1e9:.1f} GB × 利用率 {config.gpu_memory_utilization}")
        act = getattr(self, "_activation_peak", None)
        print(f"  → KV 池 {config.num_kvcache_blocks} blocks × {self.block_size} tok "
              f"({self.kv_cache.numel() * self.dtype.itemsize / 1e9:.2f} GB, "
              f"每 block {block_bytes/1e6:.2f} MB, "
              f"激活预算 {'实测 ' + f'{act/1e9:.2f}' + ' GB' if act is not None else '启发式 6 GB'})")

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
        if is_prefill:
            last = get_context().cu_seqlens_q[1:] - 1              # 每序列最后一个 q 位置
            logits = self.model.compute_logits(hidden[last])
        else:
            logits = self.model.compute_logits(hidden)
        temperatures = torch.tensor([seq.temperature for seq in seqs],
                                    dtype=torch.float32, device=self.device)
        return self.sampler(logits, temperatures).tolist()

