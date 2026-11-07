# v5/engine/model_runner.py
"""每个 GPU worker 内的模型执行器。

装载本 rank 权重、测量激活峰值、分配本 stage KV cache，
将调度结果转换为 Context，执行 forward 并在末 stage 采样。
"""

import torch

from config import Config
from engine.sequence import Sequence
from models.qwen3_moe import Qwen3MoeForCausalLM
from utils.context import set_context, get_context
from utils.loader import load_model, load_thinker_config
from engine.sampler import Sampler
from engine.intermediate_tensors import IntermediateTensors


class ModelRunner:

    def __init__(self, config: Config, dtype=None):
        self.config = config
        # A padded CUDA Graph changes the GEMM/collective shapes.  In the
        # ordinary BF16 path that is allowed to change the reduction tree and
        # therefore the logits of the real rows.  Graph mode is a correctness
        # feature by default: select the fixed-tile/fixed-reduction policy
        # before warmup, so activation sizing and graph capture see the same
        # numerical path.  Set VLLM_LEARN_DECODE_GRAPH_STRICT=0 only for an
        # explicitly requested fast (shape-sensitive) experiment.
        import os as _os
        self._decode_graph_requested = _os.getenv('VLLM_LEARN_DECODE_GRAPH', '0') == '1'
        self._decode_graph_strict = (
            self._decode_graph_requested
            and (_os.getenv('VLLM_LEARN_DECODE_GRAPH_STRICT', '1') != '0'
                 or _os.getenv('VLLM_LEARN_BATCH_INVARIANT', '0') == '1'))
        if self._decode_graph_strict:
            _os.environ['VLLM_LEARN_BATCH_INVARIANT'] = '1'
        # thinker 的 text_config 是嵌套 dict → 平铺成 SimpleNamespace（模型文件按属性读）
        from types import SimpleNamespace
        t = load_thinker_config(config.model)
        hf = SimpleNamespace(**t)
        hf.dtype = getattr(hf, "dtype", None) or torch.bfloat16
        config.hf_config = hf
        # ---- 多模态(可选,env VLLM_LEARN_MM=1):thinker vision_config ----
        if _os.getenv('VLLM_LEARN_MM') == '1':
            import json as _json
            raw = _json.load(open(f'{config.model}/config.json', encoding='utf-8'))['thinker_config']
            vcfg = dict(raw['vision_config'])
            # num_position_embeddings  config 缺省:从 checkpoint header 读 pos_embed 行数
            if 'num_position_embeddings' not in vcfg:
                from safetensors import safe_open as _so
                idx = _json.load(open(f'{config.model}/model.safetensors.index.json', encoding='utf-8'))
                fn = f"{config.model}/{idx['weight_map']['thinker.visual.pos_embed.weight']}"
                with _so(fn, framework='pt') as _f:
                    vcfg['num_position_embeddings'] = _f.get_slice('thinker.visual.pos_embed.weight').get_shape()[0]
            vcfg.setdefault('in_channels', 3)
            vcfg.setdefault('hidden_act', 'gelu_pytorch_tanh')
            hf.vision_cfg = vcfg
            hf.image_token_id = raw.get('image_token_id')
            hf.video_token_id = raw.get('video_token_id')
            self.image_token_id = hf.image_token_id
        else:
            hf.vision_cfg = None
            self.image_token_id = None
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
        self.decode_graphs = {}
        self._graph_pool = None
        self._warming_up = True
        try:
            self._warmup_model()
        finally:
            self._warming_up = False
        self.allocate_kv_cache()
        self._setup_decode_graphs()

    def shutdown(self):
        """Release request-scoped auxiliary resources."""
        if hasattr(self, '_vision_features'):
            self._vision_features.clear()
        self._release_decode_graphs()

    def _release_decode_graphs(self):
        """Destroy graph executables after all previously enqueued replays.

        Keeping this in one place matters for a failed multi-bucket capture:
        a partially populated graph pool must not survive into shutdown or a
        retry and make the next capture account for stale device allocations.
        """
        graphs = getattr(self, 'decode_graphs', None)
        if not graphs:
            self._graph_pool = None
            return
        torch.cuda.synchronize()
        for state in graphs.values():
            state['graph'].reset()
        graphs.clear()
        self._graph_pool = None

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
            seq.block_size = bs
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
        # Text warmup excludes image encoder activations and graph pools.
        if getattr(self.model.model, 'vision_tower', None) is not None:
            budget -= config.mm_feature_cache_bytes + config.mm_encoder_reserve_bytes
        import os
        graph_requested = os.getenv('VLLM_LEARN_DECODE_GRAPH', '0') == '1'
        if graph_requested:
            budget -= config.decode_graph_reserve_bytes
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
        # Capture and padded rows read a dedicated zero block inaccessible
        # to the scheduler; slots=-1 prevent writes to live KV.
        self.kv_cache = torch.empty(2, _n, config.num_kvcache_blocks + int(graph_requested),
                                    self.block_size, num_kv_heads, head_dim,
                                    dtype=self.dtype, device=self.device)
        if graph_requested:
            self.kv_cache[:, :, -1].zero_()
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

    @torch.inference_mode()
    def _build_mm_batch(self, seqs):
        """Encode complete images, then slice merged/deepstack features.

        Vision attention is bidirectional inside an image. Splitting raw
        patches changes semantics; only the language-model tokens may chunk.
        """
        if getattr(self.model.model, 'vision_tower', None) is None:
            if any((getattr(s, 'mm', None) or {}).get('images') for s in seqs):
                raise ValueError('image input requires VLLM_LEARN_MM=1 and vision weights')
            return None
        from engine.vision_cache import VisionFeatureCache
        if not hasattr(self, '_vision_features'):
            self._vision_features = VisionFeatureCache(self.config.mm_feature_cache_bytes)
        merged_rows, deep_rows, seg_pos = [], None, []
        flat_off = 0
        for seq in seqs:
            win0 = seq.num_cached_tokens
            wlen = seq.num_scheduled_tokens
            mm = getattr(seq, 'mm', None)
            for image_id, img in enumerate((mm or {}).get('images', [])):
                t, gh, gw = img['grid']
                llm_len = t * (gh // 2) * (gw // 2)
                p0 = int(img['pos'])
                lo, hi = max(p0, win0), min(p0 + llm_len, win0 + wlen)
                if hi <= lo:
                    continue
                if min(t, gh, gw) <= 0 or gh % 2 or gw % 2:
                    raise ValueError('image grid must be positive with even spatial dimensions')
                key = (seq.seq_id, image_id)
                features = self._vision_features.get(key)
                if features is None:
                    pix = torch.as_tensor(img['pixels'])
                    if pix.ndim != 2 or pix.shape[0] != t * gh * gw:
                        raise ValueError('pixel rows must match the COMPLETE image grid')
                    grid = torch.tensor([[t, gh, gw]], dtype=torch.int64)
                    features = self.model.run_vision(pix.to(self.device, torch.float32), grid)
                    if features[0].shape[0] != llm_len or any(d.shape != features[0].shape for d in features[1]):
                        raise ValueError('vision output disagrees with image span')
                    if hi < p0 + llm_len:
                        self._vision_features.put(key, features)
                merged, deepstack = features
                if deep_rows is None:
                    deep_rows = [[] for _ in deepstack]
                if len(deep_rows) != len(deepstack):
                    raise ValueError('inconsistent deepstack feature count')
                a, b = lo - p0, hi - p0
                merged_rows.append(merged[a:b])
                for rows, feature in zip(deep_rows, deepstack):
                    rows.append(feature[a:b])
                if hi == p0 + llm_len:
                    self._vision_features.pop(key)
                base = flat_off + (lo - win0)
                seg_pos.extend(range(base, base + (hi - lo)))
            flat_off += wlen
        cs = self._mrope_rows(seqs)
        if not merged_rows:
            return (None, [], None, cs) if cs is not None else None
        merged = torch.cat(merged_rows)
        ds_list = [torch.cat(r) for r in deep_rows]
        spos = torch.tensor(seg_pos, dtype=torch.int64, device=self.device)
        assert merged.size(0) == spos.numel(), (merged.shape, spos.shape)
        return merged, ds_list, spos, cs

    def _mrope_rows(self, seqs):
        """prefill 批的 mrope table [T, 2*(hd/2)]([cos64|sin64] 半布局)。

        HF recomposition_frequencies 语义:freqs 从 T 轴铺满 64 个频率,
        H 轴占 idx 1,4,...,3*sec1-2(步长 3),W 轴占 idx 2,5,...,3*sec2-1,
        其余保持 T;最终 cos=cat(freqs,freqs)(两半全等)→ 与现有
        table-kernel 的 [cos|sin] 半布局完全兼容:此处构造等价的
        [T, hd] 行,positions 用 arange(T) 查它(由 Attention 负责)。
        无非 mm 批返回 None(走原 kernel)。attention_scaling=1.0。"""
        if not any(getattr(s, 'mm', None) for s in seqs):
            return None
        import numpy as _np
        import torch as _t
        from layers.mm_preprocess import mrope_position_window
        rows = []
        for s in seqs:
            win0, wlen = s.num_cached_tokens, s.num_scheduled_tokens
            mm = getattr(s, 'mm', None)
            p3 = mrope_position_window((mm or {}).get('pos3'), win0, wlen,
                                      (mm or {}).get('delta', 0))
            rows.append(p3)
        pos3 = _t.from_numpy(_np.concatenate(rows, 1)).to(self.device, _t.float32)  # [3,T]
        T = pos3.shape[1]
        hd = self.model.model.layers[self.model.model.pp_start].self_attn.head_dim
        theta = getattr(self.config.hf_config, 'rope_theta', 10000.0)
        sec = (getattr(self.config.hf_config, 'rope_parameters', None) or {}).get(
            'mrope_section', [24, 20, 20])
        inv = 1.0 / (theta ** (_t.arange(0, hd, 2, dtype=_t.float32, device=self.device) / hd))
        fr = pos3[:, :, None] * inv[None, None, :]      # [3,T,hd/2]
        hidx = _t.arange(1, sec[1] * 3, 3, device=self.device)
        widx = _t.arange(2, sec[2] * 3, 3, device=self.device)
        freqs = fr[0].clone()                            # 先铺 T 轴
        freqs[:, hidx] = fr[1][:, hidx]                  # H 轴交错
        freqs[:, widx] = fr[2][:, widx]                  # W 轴交错
        table = _t.cat([_t.cos(freqs), _t.sin(freqs)], dim=-1)   # [T, hd] fp32
        # HF rotary 返回 cos/sin.to(x.dtype=bf16),q*cos 的乘子是 bf16 舍入值;
        # 这里同步舍入,尽力对齐 HF 的乘积舍入(差一二次舍入在近平局步仍会翻,
        # 属 bf16 数值噪声范围)
        return table.to(_t.bfloat16).contiguous()

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids, positions, slot_mapping, context_lens = [], [], [], []
        bs = self.block_size
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1 + ((getattr(seq, 'mm', None) or {}).get('delta', 0)))
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

    # ================= decode CUDA graph(可选,env VLLM_LEARN_DECODE_GRAPH=1) =================

    def _decode_graph_supported(self):
        """Only admit a genuinely static decode path; fail explicitly."""
        import os as _os
        if _os.getenv('VLLM_LEARN_DECODE_GRAPH', '0') != '1':
            return False, 'unset'
        from layers.parallel import STATE
        if getattr(STATE, 'pp_size', 1) > 1:
            return False, 'pp>1'
        if getattr(STATE, 'cp_size', 1) > 1:
            return False, 'dcp'
        if not STATE.enable_expert_parallel or STATE.ep_size <= 1:
            return False, 'requires dense EP decode; sparse expert offsets are dynamic'
        if STATE.enable_expert_parallel and getattr(STATE, 'ep_size', 1) > 1:
            # EP 常规 dispatch(a2a split 逐值动态)不能进图;dense-EP decode
            # 变体(免 dispatch 定形 GEMM)可以,但要求 dp=1
            if _os.getenv('VLLM_LEARN_EP_DENSE_DECODE', '0') != '1' or \
                    getattr(STATE, 'dp_size', 1) != 1:
                return False, 'ep alltoallv dynamic shapes'
        if self.config.batch_execution_mode == 'per_request':
            return False, 'per_request'
        from layers import kernels as _k
        if _k._REFERENCE_QK:
            return False, 'reference qk (data-dependent shapes); set VLLM_LEARN_REFERENCE_QK=0'
        from layers.attention import Attention
        from layers.moe import GroupedExperts, MoEMLP
        for m in self.model.modules():
            if isinstance(m, Attention) and m.backend != 'triton':
                return False, 'requires VLLM_LEARN_ATTENTION_BACKEND=triton'
            if isinstance(m, MoEMLP) and not m.ep_dense_decode:
                return False, 'requires VLLM_LEARN_EP_DENSE_DECODE=1'
            if isinstance(m, MoEMLP) and m.dense_backend != 'bmm':
                return False, 'dense decode graph requires DENSE_EXPERT_BACKEND=bmm'
            if isinstance(m, GroupedExperts):
                if m.quantized or m.precision != 'model':
                    return False, 'requires non-quantized model-precision experts'
        return True, ''

    def _setup_decode_graphs(self):
        ok, why = self._decode_graph_supported()
        if not ok:
            self._graph_disabled_reason = why
            if why != 'unset':
                raise ValueError(f'CUDA graph requested but unsupported: {why}')
            return
        self.graph_replay_count = 0
        self.graph_capture_count = 0
        self._graph_dummy_block = self.config.num_kvcache_blocks
        if self._graph_dummy_block != self.kv_cache.shape[2] - 1:
            raise RuntimeError('decode graph dummy KV block is outside the allocated cache')
        cap = self.config.max_num_seqs
        if cap < 1:
            raise ValueError('CUDA graph requires max_num_seqs >= 1')
        buckets = sorted({b for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
                          if b <= cap} | {cap})
        max_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        if max_blocks < 1:
            raise ValueError('CUDA graph requires max_model_len >= 1')
        # The KV budget subtracts decode_graph_reserve_bytes before allocation.
        # Measure the actual persistent graph footprint as well: a too-small
        # reserve must fail with an actionable error instead of surfacing later
        # as an unrelated allocator OOM during the last bucket.
        torch.cuda.synchronize()
        baseline_alloc = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        try:
            for b in buckets:
                self._graph_barrier()
                self._capture_decode_graph(b, max_blocks)
                torch.cuda.synchronize()
                self._graph_barrier()
                self.graph_capture_count += 1
            graph_alloc = max(0, torch.cuda.memory_allocated() - baseline_alloc)
            graph_reserved = max(0, torch.cuda.memory_reserved() - baseline_reserved)
            self._graph_capture_bytes = max(graph_alloc, graph_reserved)
            reserve = int(self.config.decode_graph_reserve_bytes)
            if self._graph_capture_bytes > reserve:
                raise RuntimeError(
                    'captured CUDA Graph pool exceeds decode_graph_reserve_bytes: '
                    f'actual={self._graph_capture_bytes / 2**30:.2f} GiB, '
                    f'reserve={reserve / 2**30:.2f} GiB; increase the reserve or '
                    'reduce max_num_seqs/max_model_len')
        except RuntimeError as exc:
            self._release_decode_graphs()
            raise RuntimeError(
                'CUDA Graph capture failed; no eager fallback is permitted for a '
                f'requested graph run ({exc})') from exc
        ranks = getattr(__import__('layers.parallel', fromlist=['STATE']), 'STATE')
        if getattr(ranks, 'tp_rank', 0) == 0:
            print(f'[decode-graph] captured buckets {sorted(self.decode_graphs)} '
                  f'(strict={self._decode_graph_strict}, '
                  f'pool={self._graph_capture_bytes / 2**20:.1f} MiB)')

    def _graph_barrier(self):
        """Keep NCCL graph capture order identical on every TP/EP rank."""
        import torch.distributed as dist
        from layers.parallel import STATE
        group = getattr(STATE, 'ep_group', None) or getattr(STATE, 'tp_group', None)
        if group is not None and dist.is_available() and dist.is_initialized():
            if dist.get_world_size(group) > 1:
                dist.barrier(group=group)

    @torch.inference_mode()
    def _capture_decode_graph(self, B, max_blocks):
        """静态输入缓冲(id/pos/slot/lens/bt)+ 一次模型前向 + logits 入图。
        slot=-1:fused kernel 的 keep mask 跳过 KV 写,capture/warmup 不污染池;
        decode_attn 对 len=1 只读专用零块,不会访问有效请求的 KV。"""
        if B < 1 or max_blocks < 1:
            raise ValueError('decode graph shape must be positive')
        dev = self.device
        ids = torch.zeros(B, dtype=torch.int64, device=dev)
        pos = torch.zeros(B, dtype=torch.int64, device=dev)
        slots = torch.full((B,), -1, dtype=torch.int32, device=dev)
        lens = torch.ones(B, dtype=torch.int32, device=dev)
        bt = torch.full((B, max_blocks), self._graph_dummy_block, dtype=torch.int32, device=dev)
        previous = get_context()
        set_context(False, slot_mapping=slots, context_lens=lens, block_tables=bt)
        g = None
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    h = self.model(ids, pos)
                    _ = self.model.compute_logits(h)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            self._graph_barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._graph_pool):
                h = self.model(ids, pos)
                logits = self.model.compute_logits(h)
        finally:
            set_context(**{name: getattr(previous, name) for name in previous.__dataclass_fields__})
        if g is None:
            raise RuntimeError(f'CUDA Graph capture produced no graph for bucket {B}')
        self._graph_pool = g.pool()
        self.decode_graphs[B] = dict(graph=g, ids=ids, pos=pos, slots=slots,
                                     lens=lens, bt=bt, logits=logits)

    def _pick_decode_graph(self, n):
        # Padding is a normal graph execution shape. Compare replay against
        # eager at THIS shape; a different M may change BF16 GEMM/collective
        # rounding in either execution mode. Do not silently fall back to
        # eager just because a batch is not itself a captured bucket.
        return min((b for b in self.decode_graphs if b >= n), default=None)

    @torch.inference_mode()
    def _replay_decode_graph(self, st, seqs):
        """把本步 decode 元数据填进静态缓冲 → replay → 返回 logits[:B](视图)。"""
        B, bs = len(seqs), self.block_size
        if B == 0:
            return st['logits'][:0]
        gb = int(st['ids'].shape[0])
        if B > gb:
            raise ValueError(f'decode batch {B} exceeds captured bucket {gb}')
        max_blocks = int(st['bt'].shape[1])
        dummy = int(self._graph_dummy_block)
        ids_l, pos_l, slot_l, len_l = [], [], [], []
        for seq in seqs:
            ids_l.append(seq.last_token)
            pos_l.append(len(seq) - 1 + ((getattr(seq, 'mm', None) or {}).get('delta', 0)))
            if len(seq) > self.config.max_model_len:
                raise ValueError('decode length exceeds captured context capacity')
            if len(seq) < 1:
                raise ValueError('decode sequence must contain at least one token')
            needed_blocks = (len(seq) + bs - 1) // bs
            if len(seq.block_table) < needed_blocks or len(seq.block_table) > max_blocks:
                raise ValueError('sequence block table does not match captured capacity')
            used_blocks = seq.block_table[:needed_blocks]
            if any(int(block) < 0 or int(block) >= dummy for block in seq.block_table):
                raise ValueError('sequence block table references the reserved graph dummy block')
            len_l.append(len(seq))
            slot_l.append(used_blocks[-1] * bs + seq.last_block_num_tokens - 1)
        ids_l += [0] * (gb - B); pos_l += [0] * (gb - B)
        slot_l += [-1] * (gb - B); len_l += [1] * (gb - B)
        st['ids'].copy_(torch.tensor(ids_l, dtype=torch.int64), non_blocking=True)
        st['pos'].copy_(torch.tensor(pos_l, dtype=torch.int64), non_blocking=True)
        st['slots'].copy_(torch.tensor(slot_l, dtype=torch.int32), non_blocking=True)
        st['lens'].copy_(torch.tensor(len_l, dtype=torch.int32), non_blocking=True)
        rows = [list(seq.block_table) + [-1] * (max_blocks - len(seq.block_table))
                for seq in seqs]
        rows += [[dummy] * max_blocks for _ in range(gb - B)]
        st['bt'].copy_(torch.tensor(rows, dtype=torch.int32), non_blocking=True)
        # Copies and replay execute in order on the same CUDA stream.
        st['graph'].replay()
        self.graph_replay_count += 1
        return st['logits'][:B]

    # ================= 前向 + 采样 =================

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool,
            microbatch_id: int = 0) -> list[int]:
        # Numerical reproducibility mode: keep each request's GEMM M shape
        # equal to its single-request path. Every PP worker receives the same
        # ordered sequence list and applies the same split. This intentionally
        # sacrifices batch fusion and remains opt-in.
        from layers.parallel import STATE
        if not seqs:
            if not STATE.enable_expert_parallel:
                return []
            # Idle DP replicas must service other replicas' expert routes.
            # TP attention and PP activations have no local tokens; skipping
            # them is symmetric within this DP replica, but skipping MoE
            # collectives across DP would deadlock active replicas.
            x = torch.empty((0, self.config.hf_config.hidden_size),
                            device=self.device, dtype=self.dtype)
            for layer in self.model.model.layers[self.model.model.pp_start:self.model.model.pp_end]:
                layer.mlp(x)
            return []
        if self.config.batch_execution_mode == "per_request" and len(seqs) > 1:
            result = []
            for seq in seqs:
                result.extend(self.run([seq], is_prefill, microbatch_id))
            return result
        # A mixed scheduler batch must not change an existing decode request
        # into the varlen attention / sparse-EP prefill numerical path. Keep
        # its decode kernels (and Graph replay) even when new prompts arrive.
        # All TP peers see the same split. DP/PP/CP need a separate protocol
        # for subdividing messages and are deliberately excluded here.
        from layers.batch_invariant import enabled as batch_invariant
        if (is_prefill and batch_invariant()
                and STATE.dp_size == STATE.pp_size == STATE.cp_size == 1):
            decode = [i for i, seq in enumerate(seqs)
                      if seq.num_cached_tokens >= seq.num_prompt_tokens]
            if decode and len(decode) < len(seqs):
                decode_set = set(decode)
                prefill = [i for i in range(len(seqs)) if i not in decode_set]
                result = [0] * len(seqs)
                for indices, phase in ((decode, False), (prefill, True)):
                    values = self.run([seqs[i] for i in indices], phase, microbatch_id)
                    for i, value in zip(indices, values):
                        result[i] = value
                return result
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
            mm_batch = self._build_mm_batch(seqs)
        else:
            mm_batch = None
            b = self._pick_decode_graph(len(seqs)) if getattr(self, 'decode_graphs', None) else None
            if b is not None:
                st = self.decode_graphs[b]
                logits = self._replay_decode_graph(st, seqs)
                temperatures = torch.tensor([seq.temperature for seq in seqs],
                                            dtype=torch.float32, device=self.device)
                return self.sampler(logits, temperatures).tolist()
            if (getattr(self, '_decode_graph_requested', False)
                    and not getattr(self, '_warming_up', False)):
                # A requested graph run must never silently turn into eager
                # because a caller exceeded the captured capacity.  Eager is
                # still available by leaving VLLM_LEARN_DECODE_GRAPH unset.
                raise RuntimeError(
                    f'decode batch {len(seqs)} has no captured CUDA Graph bucket '
                    f'(max={max(getattr(self, "decode_graphs", {}), default=0)})')
            input_ids, positions = self.prepare_decode(seqs)
        intermediate = None
        transport = STATE.pipeline_transport
        if STATE.pp_size > 1 and not STATE.pipeline_warmup and STATE.pp_rank > 0:
            if transport is None:
                raise RuntimeError("PP transport was not initialized")
            hidden, residual = transport.recv_forward(
                microbatch_id=microbatch_id,
                shape=(input_ids.numel(), self.config.hf_config.hidden_size),
                dtype=self.dtype)
            intermediate = IntermediateTensors(hidden, residual)
        hidden = self.model(input_ids, positions, intermediate_tensors=intermediate,
                            mm_batch=mm_batch)
        # Only the execution layer sends activations. Model forward has no
        # process-group/transport side effects and returns an explicit boundary.
        if getattr(STATE, "pp_size", 1) > 1 and STATE.pp_rank < STATE.pp_size - 1:
            if not isinstance(hidden, IntermediateTensors):
                raise RuntimeError("non-last PP stage must return IntermediateTensors")
            if not STATE.pipeline_warmup:
                if transport is None:
                    raise RuntimeError("PP transport was not initialized")
                transport.send_forward(hidden.hidden_states, hidden.residual,
                                       microbatch_id=microbatch_id)
            return [0] * len(seqs)
        if is_prefill:
            last = get_context().cu_seqlens_q[1:] - 1              # 每序列最后一个 q 位置
            logits = self.model.compute_logits(hidden[last])
        else:
            logits = self.model.compute_logits(hidden)
        temperatures = torch.tensor([seq.temperature for seq in seqs],
                                    dtype=torch.float32, device=self.device)
        return self.sampler(logits, temperatures).tolist()
