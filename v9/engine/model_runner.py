# v5/engine/model_runner.py
"""每个 GPU worker 内的模型执行器。

装载本 rank 权重、测量激活峰值、分配本 stage KV cache，
将调度结果转换为 Context，执行 forward 并在末 stage 采样。
"""
import os

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

        # Qwen3-Omni-30B thinker：CPU 构造 + 按当前 TP/EP/PP rank 直接装载
        # 本 rank 权重。模型、loader 和运行时保持单一协议，避免按模型名劫持
        # forward 或在 GPU 上短暂出现全量权重。
        self.model = Qwen3MoeForCausalLM(hf, config.max_model_len)
        load_model(self.model, config.model, device=self.device, dtype=self.dtype)
        self.model.eval()
        self.sampler = Sampler()

        # ---- warmup 量激活峰值 → 按显存预算公式定池子 ----
        # v9.3 组件化并行：模型构造时就只有本 rank 的 1/N 权重（不再有
        # "全量加载再切分"的阶段）——warmup 可以在任何配置下直接跑。
        # launcher 责任：构造 ModelRunner 之前 configure_parallel()
        self.kv_cache = None
        self._warmup_model()
        self.allocate_kv_cache()
        # Diagnostics are enabled only after memory-sizing warmup. Capturing
        # CPU logits synchronizes the device and must never be used for speed.
        self._logit_capture_dir = os.getenv("VLLM_LEARN_LOGITS_DIR")
        self._logit_capture_count = 0
        self._logit_capture_limit = int(os.getenv("VLLM_LEARN_LOGITS_LIMIT", "128"))
        capture_dir = os.getenv("VLLM_LEARN_ACTIVATIONS_DIR")
        if capture_dir:
            from layers.parallel import STATE
            if STATE.tp_rank == STATE.ep_rank == STATE.cp_rank == 0:
                from utils.activation_capture import install_activation_capture
                self._activation_handles = install_activation_capture(
                    self.model, capture_dir, STATE.pp_rank)

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
        from layers.parallel import STATE
        n_local = self.config.hf_config.num_hidden_layers
        if getattr(STATE, "pp_size", 1) > 1:
            from layers.parallel import get_pp_indices
            s, e = get_pp_indices(n_local, STATE.pp_rank, STATE.pp_size)
            n_local = e - s
        assert layer_id == n_local, f"{layer_id} != 本 stage 层数 {n_local}"

    def _per_rank_kv_heads(self):
        """per-rank kv 头数：从模型第一层 attention 组件读（v9.3 组件构造时
        已按 ParallelState 切——TP2 下 kv=4 → 每 rank 2；kv 复制时 ≥1）。
        引擎读取组件已经确定的头数，不在装载后再次修改模型。"""
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
        n_layers = hf.num_hidden_layers
        from layers.parallel import STATE as _ST
        if getattr(_ST, "pp_size", 1) > 1:
            from layers.parallel import get_pp_indices
            _s, _e = get_pp_indices(n_layers, _ST.pp_rank, _ST.pp_size)
            n_layers = _e - _s
        hd = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        blocks_per_seq = (seq_len + bs - 1) // bs
        tmp = torch.empty(2, n_layers, num_seqs * blocks_per_seq, bs, kvh, hd,
                          dtype=self.dtype, device=self.device)
        self._assign_layer_caches(tmp)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for i, seq in enumerate(seqs):
            seq.block_table = [i * blocks_per_seq + j for j in range(blocks_per_seq)]
            seq.num_scheduled_tokens = seq_len
        from layers.parallel import STATE as _warmup_state
        _warmup_state.pipeline_warmup = getattr(_warmup_state, "pp_size", 1) > 1
        self.run(seqs, True)
        for seq in seqs:                              # decode 形状
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
        self.run(seqs, False)
        _warmup_state.pipeline_warmup = False
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
        num_kv_heads = self._per_rank_kv_heads()
        head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        from layers.parallel import STATE as _ST2
        _n = hf.num_hidden_layers
        if getattr(_ST2, "pp_size", 1) > 1:
            from layers.parallel import get_pp_indices
            _s2, _e2 = get_pp_indices(_n, _ST2.pp_rank, _ST2.pp_size)
            _n = _e2 - _s2
        block_bytes = 2 * _n * self.block_size * num_kv_heads * head_dim * self.dtype.itemsize
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
        if config.max_kvcache_blocks is not None:
            config.num_kvcache_blocks = min(config.num_kvcache_blocks, config.max_kvcache_blocks)
        if config.num_kvcache_blocks <= 0:
            raise RuntimeError(
                f'KV cache budget exhausted: total={total/1e9:.2f} GB, '
                f'used={used/1e9:.2f} GB, model_allocated={current/1e9:.2f} GB, '
                f'budget={budget/1e9:.2f} GB, block={block_bytes/1e6:.2f} MB. '
                'Reduce memory demand or choose a device with more free memory; '
                'gpu_memory_utilization applies to total device memory, including other processes.')
        self.kv_cache = torch.empty(2, _n, config.num_kvcache_blocks,
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

    def flush_pipeline(self):
        """Drain asynchronous PP sends before the process exits."""
        transport = getattr(getattr(self.model, "model", None),
                            "pipeline_transport", None)
        if transport is not None:
            transport.flush()

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool,
            pipeline_lane: int = 0, microbatch_id: int = 0) -> list[int]:
        # Numerical reproducibility mode: keep each request's GEMM M shape
        # equal to its single-request path. Every PP worker receives the same
        # ordered sequence list and applies the same split. This intentionally
        # sacrifices batch fusion and remains opt-in.
        from layers.parallel import STATE
        if self.config.batch_execution_mode == "per_request" and len(seqs) > 1:
            result = []
            for seq in seqs:
                result.extend(self.run([seq], is_prefill, pipeline_lane,
                                      microbatch_id))
            return result
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
        # lane/微批元数据只用于 PP transport 的调度和 trace，不改变模型数学。
        STATE.pipeline_lane = pipeline_lane
        STATE.microbatch_id = microbatch_id
        with torch.autograd.profiler.record_function(
                f"stage{STATE.pp_rank}.microbatch{microbatch_id}.lane{pipeline_lane}."
                f"{'prefill' if is_prefill else 'decode'}"):
            hidden = self.model(input_ids, positions)             # [total_q, hidden]
        # PP 中间 stage：hidden 已发给下游，跳过 lm_head/采样（worker 只在
        # last stage 发 response）
        if getattr(STATE, "pp_size", 1) > 1 and STATE.pp_rank < STATE.pp_size - 1:
            return [0] * len(seqs)
        if is_prefill:
            last = get_context().cu_seqlens_q[1:] - 1              # 每序列最后一个 q 位置
            logits = self.model.compute_logits(hidden[last])
        else:
            logits = self.model.compute_logits(hidden)
        if getattr(self, "_logit_capture_dir", None):
            self._capture_logits(logits, seqs, is_prefill)
        temperatures = torch.tensor([seq.temperature for seq in seqs],
                                    dtype=torch.float32, device=self.device)
        return self.sampler(logits, temperatures).tolist()

    def _capture_logits(self, logits, seqs, is_prefill):
        """Save terminal-query rows on one final-stage replica, for oracles."""
        from pathlib import Path
        from layers.parallel import STATE
        if STATE.ep_rank or STATE.tp_rank or getattr(STATE, "cp_rank", 0):
            return
        if self._logit_capture_count >= self._logit_capture_limit:
            raise RuntimeError("Diagnostic logits limit exceeded; reduce probe size")
        selected = [i for i, seq in enumerate(seqs)
                    if seq.num_cached_tokens + seq.num_scheduled_tokens >= len(seq)]
        if not selected:
            return
        path = Path(self._logit_capture_dir)
        path.mkdir(parents=True, exist_ok=True)
        record = dict(logits=logits[selected].float().cpu(),
                      seq_ids=[seqs[i].seq_id for i in selected],
                      prompt_lengths=[len(seqs[i]) for i in selected],
                      pp_rank=STATE.pp_rank, is_prefill=is_prefill,
                      measurement="synchronizing numerical diagnostic; no throughput")
        filename = path / f"logits_stage{STATE.pp_rank}_{self._logit_capture_count:04d}.pt"
        torch.save(record, filename)
        self._logit_capture_count += 1

