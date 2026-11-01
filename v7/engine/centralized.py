# v6/engine/centralized.py
"""v6.4 调度集中化（哑 forward 架构）。

上一版的教训（MILESTONES §6.4）：follower 镜像本地 Sequence 状态的字段
永远列不全（num_cached_tokens 漏了 → 每步重算全序列），变长张量是
NCCL 序列错位之源。

本版的设计：
  1. **固定大小计划张量**——预分配 [MAX_SEQS, 6] + [MAX_SEQS, MAX_BLOCKS]，
     每步原地填充广播，永不重新分配 → 通信形状恒定，无序列错位；
  2. **follower 哑 forward**——不维护任何本地 Sequence 状态，每步从计划
     张量直接构造 forward 输入（用完即弃的输入快照，不是对象镜像）；
  3. master 走完全正常的 scheduler + postprocess（与单卡同路径）。

计划张量每行 6 个字段：
  [seq_id, is_prefill, num_scheduled_tokens, num_cached_tokens,
   bt_row, bt_len]；块表本体放 bt_buf 的对应行。
"""
import os

import torch
import torch.distributed as dist

MAX_SEQS = 8          # 与 scheduler 的 max_num_seqs 一致
MAX_BLOCKS = 40        # 单请求最大块数（256 tok/块 → 10K token 上限）
MAX_TOKENS = 16384     # 全部序列本步 token 的总预算（chunked prefill 兜底）


def _world_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


class _SeqSpec:
    """一次 forward 的输入描述（哑 forward 的『临时变量』，不是状态）。
    tokens 是本步要 forward 的 token 列表（prefill: 待算的 prompt 段；
    decode: 单个 last_token）。"""

    def __init__(self, seq_id, num_cached, num_sched, block_table, tokens,
                 last_token=None, seq_len=None):
        self.seq_id = seq_id
        self.num_prompt_tokens = num_cached + num_sched   # 哑视角下够用即可
        self.num_cached_tokens = num_cached
        self.num_scheduled_tokens = num_sched
        self.block_table = list(block_table)
        self.temperature = 0.0     # follower 的采样被 master 广播覆盖，值无所谓
        self._tokens = list(tokens)
        # decode 路径需要：
        self.last_token = tokens[0] if last_token is None else last_token
        self._seq_len = seq_len if seq_len is not None else num_cached + num_sched
        # 尾块内 token 数 = 全长 - 已满块数×块大小
        bs = block_table and max(1, self._seq_len // max(len(block_table), 1)) or 1
        self.last_block_num_tokens = self._seq_len - (len(block_table) - 1) * 256 \
            if block_table else self._seq_len

    def __len__(self):
        return self._seq_len

    def __getitem__(self, key):
        return self._tokens[key]


class CentralizedEngine:
    """master 全权调度；follower 只跑 forward。"""

    def __init__(self, engine, master_rank: int = 0):
        self.engine = engine
        self.master = master_rank
        self.is_master = (not _world_active()) or dist.get_rank() == master_rank
        # 固定大小的通信缓冲（每步原地填充——形状恒定，无变长死锁）
        self.plan_buf = torch.zeros(MAX_SEQS, 6, dtype=torch.long, device="cuda")
        self.bt_buf = torch.zeros(MAX_SEQS, MAX_BLOCKS, dtype=torch.long, device="cuda")
        self.tok_buf = torch.zeros(MAX_TOKENS, dtype=torch.long, device="cuda")
        self._last_tokens = {}     # seq_id → 上一步 token（decode 输入）
        self._master_done = False

    # ---- master: 打包 ----

    def pack(self, seqs, is_prefill: bool):
        self.plan_buf.zero_()
        self.bt_buf.zero_()
        self.tok_buf.zero_()
        n = min(len(seqs), MAX_SEQS)
        # 空行哨兵 = -1（不能用 0——seq_id 从 0 编号会与 zero_() 冲突，
        # follower 的 ne(0) 计数会漏掉第一条序列 → ALLREDUCE 尺寸错位）
        for i in range(n, MAX_SEQS):
            self.plan_buf[i, 0] = -1
        tok_off = 0
        for i, s in enumerate(seqs[:n]):
            sched = s.num_scheduled_tokens
            self.plan_buf[i, 0] = s.seq_id
            self.plan_buf[i, 1] = int(is_prefill)
            self.plan_buf[i, 2] = sched
            self.plan_buf[i, 3] = s.num_cached_tokens
            self.plan_buf[i, 4] = i
            self.plan_buf[i, 5] = len(s.block_table)
            bt = s.block_table[:MAX_BLOCKS]
            if bt:
                self.bt_buf[i, :len(bt)] = torch.tensor(bt, device="cuda")
            # 本步要 forward 的 token 段（prefill: 待算段; decode: last_token）
            toks = s[s.num_cached_tokens:s.num_cached_tokens + sched] if is_prefill \
                else [s.last_token]
            toks = toks[:max(MAX_TOKENS - tok_off, 0)]
            if toks:
                self.tok_buf[tok_off:tok_off + len(toks)] = torch.tensor(toks, device="cuda")
            self.plan_buf[i, 1] = (tok_off << 1) | int(is_prefill)   # 复用字段1: 位置<<1|标志
            tok_off += len(toks)

    # ---- follower: 解包成哑输入 ----

    def unpack(self):
        specs = []
        n = int(self.plan_buf[:, 0].ne(-1).sum().item())
        for i in range(n):
            sid = int(self.plan_buf[i, 0].item())
            packed = int(self.plan_buf[i, 1].item())
            tok_off, is_pf = packed >> 1, packed & 1
            n_sched = int(self.plan_buf[i, 2].item())
            n_cached = int(self.plan_buf[i, 3].item())
            bt_row = int(self.plan_buf[i, 4].item())
            bt_len = int(self.plan_buf[i, 5].item())
            bt = self.bt_buf[bt_row, :bt_len].tolist()
            if is_pf:
                toks = self.tok_buf[tok_off:tok_off + n_sched].tolist()
                specs.append(_SeqSpec(sid, n_cached, n_sched, bt, toks))
            else:
                toks = [self._last_tokens.get(sid, 0)]
                # decode: 序列全长 = num_cached + 1（本步新 token）
                specs.append(_SeqSpec(sid, n_cached, 1, bt, toks,
                                       seq_len=n_cached + 1))
        return specs


def make_centralized_step(engine, master_rank: int = 0):
    """替换 engine.step 为集中化哑 forward 版本。
    返回 (step, ce)。follower 的 forward 输入完全来自固定计划张量。"""
    ce = CentralizedEngine(engine, master_rank)
    sched = engine.scheduler
    _orig_is_finished = engine.is_finished

    def _is_finished():
        return _orig_is_finished() if ce.is_master else ce._master_done

    engine.is_finished = _is_finished

    def step():
        is_prefill = False
        if ce.is_master:
            seqs, is_prefill = sched.schedule()
            ce.pack(seqs, is_prefill)

        # ---- 固定形状广播（双端各三次，永不缺席 → 无序列错位）----
        # ⚠ NCCL broadcast 异步返回：发起≠完成。不 synchronize 就读 plan_buf
        # 会拿到旧缓冲（实测 follower 少看到一条序列 → ALLREDUCE 尺寸错位死锁）
        if _world_active():
            dist.broadcast(ce.plan_buf, src=ce.master)
            dist.broadcast(ce.bt_buf, src=ce.master)
            dist.broadcast(ce.tok_buf, src=ce.master)
            torch.cuda.synchronize()

        n = int(ce.plan_buf[:, 0].ne(-1).sum().item())
        if _world_active() and not ce.is_master and os.environ.get("CENT_DBG"):
            print(f"[r{dist.get_rank()}] raw plan_buf rows0-2: "
                  f"{ce.plan_buf[:3].tolist()}", flush=True)
        if _world_active() and not ce.is_master:
            ce._master_done = (n == 0)
            if n == 0:
                return []
            run_seqs = ce.unpack()
            is_prefill = bool(ce.plan_buf[0, 1].item() & 1)
        else:
            if n == 0:
                return []
            run_seqs = seqs

        # ---- forward（双端同样的输入描述）----
        import torch.distributed as _d
        _tq = sum(x.num_scheduled_tokens for x in run_seqs)
        if os.environ.get("CENT_DBG"):
            print(f"[r{_d.get_rank()}] prefill={is_prefill} total_q={_tq} "
                  f"specs={[(x.seq_id, x.num_scheduled_tokens) for x in run_seqs]}", flush=True)
        token_ids = engine.model_runner.run(run_seqs, is_prefill)

        # ---- token 广播（对齐采样）----
        t = torch.tensor(token_ids, dtype=torch.long, device="cuda")
        if _world_active():
            dist.broadcast(t, src=ce.master)
            torch.cuda.synchronize()
            token_ids = t.tolist()

        # ---- 只有 master 推进 scheduler 状态 ----
        if ce.is_master:
            sched.postprocess(seqs, token_ids, is_prefill)

        # last_token 流水（follower 下步 decode 的输入）
        for s, tok in zip(run_seqs, token_ids):
            ce._last_tokens[s.seq_id] = tok

        return [(s.seq_id, getattr(s, "completion_token_ids", []))
                for s in run_seqs if getattr(s, "is_finished", False)]

    return step, ce
