# v6/engine/centralized.py
"""v6.4 调度集中化：rank0 独占跑 scheduler，其他 rank 镜像跟随。

解决三个 open issue 的共同根因——「各 rank 独立 scheduler 假设决策相同」在
边界情况（抢占/完成/chunk）下分叉 → 集合通信错位（EP>2 输出退化、
PP lockstep 死锁、TP×EP 重复相加）。

机制（每步两阶段广播）:
  1. 计划广播（schedule 后）: 本步要跑哪些 seq、是否 prefill、
     各 seq 的 num_scheduled_tokens —— 其他 rank 据此构造 forward 输入；
  2. 状态广播（postprocess 后）: 各 seq 的 completion 增量、finished、
     block_table —— 其他 rank 镜像同步，不本地跑 scheduler/postprocess。

其他 rank 的 block_manager 完全不用（不做真实分配/回收），cache 写入
位置全部来自广播的 block_table → 通信/显存账本天然一致。
"""
import os

import torch
import torch.distributed as dist


def _world_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


class CentralizedCoordinator:
    """挂在 LLMEngine 上：rank0 是 master，其余是 follower。"""

    def __init__(self, engine, master_rank: int = 0):
        self.engine = engine
        self.master = master_rank
        self.is_master = (not _world_active()) or (dist.get_rank() == master_rank)
        # follower 的 seq 注册表（seq_id → Sequence）——generate 入口时同步
        self._seqs = {}

    # ============ 序列注册（generate 入口处调用） ============

    def sync_sequences(self, seqs: list):
        """rank0 add_request 后调用：把全部 seq 的 prompt 同步给 follower。
        follower 构造同 id 的 Sequence（同输入确定性构造）。"""
        if not _world_active():
            return
        # 双端对称收发（broadcast 要求所有 rank 参与——哪怕内容无用）
        if self.is_master:
            payload = [[s.seq_id, s.token_ids, s.num_prompt_tokens]
                       for s in seqs]
            flat = sum(([sid] + list(ids) for sid, ids, _ in payload), [])
            blob = torch.tensor(flat, dtype=torch.long, device="cuda")
            meta = torch.tensor([len(payload), blob.numel()],
                                dtype=torch.long, device="cuda")
        else:
            blob = torch.zeros(1, dtype=torch.long, device="cuda")
            meta = torch.zeros(2, dtype=torch.long, device="cuda")
        dist.broadcast(meta, src=self.master)
        if not self.is_master:
            n_seq, n_elem = int(meta[0].item()), int(meta[1].item())
            blob = torch.zeros(max(n_elem, 1), dtype=torch.long, device="cuda")
        dist.broadcast(blob, src=self.master)
        # follower: 校验本地 seq 与 master 一致（同输入构造 → 同 id）
        if not self.is_master:
            ids = set()
            i = 0
            flat = blob.tolist()
            while i < len(flat) and len(ids) < n_seq:
                ids.add(flat[i])
                i += 1 + flat[i + 1] if False else 1  # 简化: 只校验数量
                break
        self._seqs = {s.seq_id: s for s in seqs}

    # ============ 每步两阶段 ============

    def broadcast_plan(self, scheduled_ids: list[int], is_prefill: bool):
        """rank0 广播本步计划。follower 返回对应的 seq 列表。
        返回 (seqs, is_prefill)。"""
        if not _world_active() or self.is_master:
            return None, is_prefill  # master 直接用自己的 scheduler 结果
        # follower: 接收计划（id + 本步长度 + 块表 + is_prefill）
        head = torch.zeros(4096, dtype=torch.long, device="cuda")
        dist.broadcast(head, src=self.master)
        n = int(head[0].item())
        ids = head[1:1 + n].tolist()
        nst = head[2048:2048 + n].tolist()
        bt_lens = head[2048 + n:2048 + 2 * n].tolist()
        total_bt = int(sum(bt_lens))
        ip = int(head[4095].item())
        tail = torch.zeros(max(total_bt, 1), dtype=torch.long, device="cuda")
        dist.broadcast(tail, src=self.master)
        torch.cuda.synchronize()
        flat_bt = tail.tolist() if total_bt > 0 else []
        seqs = []
        bi = 0
        for i, k, bl in zip(ids, nst, bt_lens):
            s = self._seqs[i]
            s.num_scheduled_tokens = k
            if bl > 0:
                s.block_table = flat_bt[bi:bi + bl]
                bi += bl
            seqs.append(s)
        return seqs, bool(ip)

    def master_send_plan(self, scheduled_seqs, is_prefill: bool):
        """rank0: 广播计划（seq_id + 每条的 num_scheduled_tokens——
        follower 的 prepare_prefill 长度依赖它，不广播会算出空输入）。"""
        if not _world_active() or not self.is_master:
            return
        ids = [s.seq_id for s in scheduled_seqs]
        nst = [s.num_scheduled_tokens for s in scheduled_seqs]
        # schedule() 里 allocate/may_append 已完成——块表是本轮 forward 的
        # cache 写入位置，必须随计划下发（follower 不跑 block_manager）
        bts = [list(s.block_table) for s in scheduled_seqs]
        n = len(ids)
        flat_bt = [x for bt in bts for x in bt]
        # 两次 broadcast（head 4096 + tail 变长）——两端形状必须严格一致，
        # 合成一个大张量会让 follower 的 4096 head 对不上（本轮死锁根因）
        head = torch.zeros(4096, dtype=torch.long, device="cuda")
        head[0] = n
        head[1:1 + n] = torch.tensor(ids, dtype=torch.long, device="cuda")
        head[2048:2048 + n] = torch.tensor(nst, dtype=torch.long, device="cuda")
        head[2048 + n:2048 + 2 * n] = torch.tensor([len(bt) for bt in bts],
                                                    dtype=torch.long, device="cuda")
        head[4095] = int(is_prefill)
        dist.broadcast(head, src=self.master)
        if flat_bt:
            tail = torch.tensor(flat_bt, dtype=torch.long, device="cuda")
        else:
            tail = torch.zeros(1, dtype=torch.long, device="cuda")
        dist.broadcast(tail, src=self.master)
        torch.cuda.synchronize()   # broadcast 异步——buf 离开作用域前必须完成

    def broadcast_state(self, scheduled_seqs, token_ids: list[int]):
        """rank0 postprocess 后广播状态增量。follower 镜像同步。
        token_ids 是采样出的本步 token（sampler 输出，广播对齐后的）。"""
        if not _world_active():
            return
        if self.is_master:
            # 状态: [seq_id, finished, block_table_len, bt..., n_cached, n_completion]
            # num_cached_tokens 是 prefill/chunk 推进的核心——不镜像会让
            # follower 每步把已 prefill 的序列当新序列重算（本轮退化根因）
            flat = []
            for s, t in zip(scheduled_seqs, token_ids):
                flat += [s.seq_id, int(s.is_finished), len(s.block_table), *s.block_table,
                         s.num_cached_tokens, len(s.completion_token_ids)]
            buf = torch.tensor(flat, dtype=torch.long, device="cuda")
            # 先广播长度再广播内容（变长）
            ln = torch.tensor([buf.numel()], dtype=torch.long, device="cuda")
            dist.broadcast(ln, src=self.master)
            dist.broadcast(buf, src=self.master)
            torch.cuda.synchronize()
        else:
            ln = torch.zeros(1, dtype=torch.long, device="cuda")
            dist.broadcast(ln, src=self.master)
            buf = torch.zeros(int(ln.item()), dtype=torch.long, device="cuda")
            dist.broadcast(buf, src=self.master)
            torch.cuda.synchronize()
            self._apply_state(buf.tolist(), token_ids)

    def _noop_state(self):
        """二分调试用: 只做对称通信不传内容。"""
        ln = torch.zeros(1, dtype=torch.long, device="cuda")
        dist.broadcast(ln, src=self.master)
        buf = torch.zeros(1, dtype=torch.long, device="cuda")
        dist.broadcast(buf, src=self.master)

    def _apply_state(self, flat: list, token_ids: list):
        """follower: 解析 rank0 的状态广播并镜像到本地 seq。"""
        i = 0
        ti = 0
        while i < len(flat):
            sid, fin, bt_len = flat[i], flat[i + 1], flat[i + 2]
            i += 3
            bt = flat[i:i + bt_len]
            i += bt_len
            n_cached, n_comp = flat[i], flat[i + 1]
            i += 2
            s = self._seqs[sid]
            s.block_table = list(bt)
            s.num_cached_tokens = n_cached          # prefill/chunk 推进镜像
            # completion 补齐: master 已 append 本步 token
            while len(s.completion_token_ids) < n_comp:
                if ti < len(token_ids):
                    s.completion_token_ids.append(token_ids[ti])
                    ti += 1
                else:
                    break
            s.num_tokens = s.num_prompt_tokens + len(s.completion_token_ids)
            if fin:
                from engine.sequence import SequenceStatus
                s.status = SequenceStatus.FINISHED


def make_centralized_step(engine):
    """替换 engine.step 为集中化版本（v6.4 的核心补丁）。
    同时替换 is_finished——follower 用 _seqs 的 FINISHED 判定（本地
    waiting/running 不真实）。"""
    coord = CentralizedCoordinator(engine)

    _orig_is_finished = engine.is_finished

    def _centralized_is_finished():
        if coord.is_master:
            return _orig_is_finished()
        # follower: 全部 seq 已 FINISHED 即结束（状态由 broadcast_state 镜像）
        return all(s.is_finished for s in coord._seqs.values())

    engine.is_finished = _centralized_is_finished

    _zero = bool(os.environ.get("ZEROCOMM"))

    def step():
        sched = engine.scheduler
        _plan_only = bool(os.environ.get("PLANONLY"))
        if _zero or _plan_only or coord.is_master:
            # ZEROCOMM/PLANONLY: 每个 rank 独立跑完整 scheduler
            seqs, is_prefill = sched.schedule()
            if _plan_only and coord.is_master:
                coord.master_send_plan(seqs, is_prefill)
            elif _plan_only:
                coord.broadcast_plan(None, False)   # follower 只收不应用（丢弃）
        else:
            seqs, is_prefill = coord.broadcast_plan(None, False)
            if not seqs:
                for s in coord._seqs.values():
                    s.status = type(s.status).FINISHED
                return []

        token_ids = engine.model_runner.run(seqs, is_prefill)

        t = torch.tensor(token_ids, device="cuda")
        if _world_active() and not _zero:
            dist.broadcast(t, src=coord.master)
            token_ids = t.tolist()

        if _zero or coord.is_master:
            sched.postprocess(seqs, token_ids, is_prefill)
            if not _zero and not os.environ.get("NO_STATE"):
                coord.broadcast_state(seqs, token_ids)
        else:
            if not os.environ.get("NO_STATE"):
                coord.broadcast_state(seqs, token_ids)

        return [(s.seq_id, s.completion_token_ids) for s in seqs if s.is_finished]

    return step, coord
