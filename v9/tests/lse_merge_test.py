"""v8.0: CP/DCP 的核心数学——partial attention + LSE 加权合并的单元验证。

问题: 序列的 KV 分到 N 卡，每卡算 partial attention（只用自己的 KV 分片），
如何无损合并成全量 attention 的结果？

答案: LSE (log-sum-exp) 加权合并——在线 softmax 数学在空间分片上的扩展：
  partial_i = softmax(qK_i^T/√d) V_i          # 各卡的局部 attention
  m_i = max(qK_i^T/√d)                         # 各卡的局部最大值（数值稳定）
  l_i = Σ exp(qK_i^T/√d - m_i)                 # 各卡的局部和
  合并: out = Σ_i [exp(m_i - m*) · l_i] / Σ_i [exp(m_i - m*) · l_i] · partial_i
  其中 m* = max(m_i)
  （等价于把各卡的 softmax 分母统一到全局最大值——与 flash attention 的
   在线重缩放同一数学，只是"流式块"变成"空间分片"）

验证: N=1/2/4/8 分片的合并结果 vs 全量 attention fp32 逐位对比。
"""
import torch


def full_attention(q, k, v, scale):
    """参考实现: [T, H, D] q, [L, H, D] k/v → [T, H, D]。fp32。"""
    scores = torch.einsum("thd,lhd->htl", q, k) * scale     # [H, T, L]
    probs = scores.softmax(-1)
    return torch.einsum("htl,lhd->thd", probs, v)


def partial_with_lse(q, k_shard, v_shard, scale):
    """单卡 partial: 返回 (partial_out, m, l)。
    m/l 是 softmax 的最大值和分母（未归一化）——合并的关键中间量。"""
    scores = torch.einsum("thd,lhd->htl", q, k_shard) * scale  # [H, T, S]
    m = scores.max(-1).values                                  # [H, T]
    p = torch.exp(scores - m[..., None])
    l = p.sum(-1)                                              # [H, T]
    out = torch.einsum("htl,lhd->thd", p / l[..., None], v_shard)
    return out, m, l


def lse_merge(partials, ms, ls):
    """LSE 加权合并 N 个 partial → 全量结果。
    权重 w_i = exp(m_i - m*) · l_i，归一化后加权 partial。
    m/l 是 [H, T]，partial 是 [T, H, D]——权重转置对齐到 [T, H, 1]。"""
    m_star = torch.stack(ms).max(0).values                     # [H, T] 全局最大
    ws = [torch.exp(m - m_star) * l for m, l in zip(ms, ls)]  # [H, T]
    w_total = sum(ws)                                          # [H, T]
    out = sum(w.T[..., None] * p for w, p in zip(ws, partials))
    return out / w_total.T[..., None]


def test(N):
    torch.manual_seed(42)
    T, L, H, D = 7, 96, 8, 64
    scale = D ** -0.5
    q = torch.randn(T, H, D)
    k = torch.randn(L, H, D)
    v = torch.randn(L, H, D)

    ref = full_attention(q, k, v, scale)

    # 序列按 interleave=1 分片（DCP 风格——位置 i 归卡 i%N）
    shards = [(k[i::N], v[i::N]) for i in range(N)]
    partials, ms, ls = [], [], []
    for ks, vs in shards:
        p, m, l = partial_with_lse(q, ks, vs, scale)
        partials.append(p)
        ms.append(m)
        ls.append(l)
    merged = lse_merge(partials, ms, ls)

    diff = (merged - ref).abs().max().item()
    print(f"N={N}: LSE 合并 vs 全量 attention, max diff = {diff:.2e}  "
          f"{'PASS' if diff < 1e-5 else 'FAIL'}")
    return diff < 1e-5


if __name__ == "__main__":
    ok = all(test(n) for n in [1, 2, 3, 4, 8])
    print("\n===== v8.0 LSE 合并数学:", "ALL PASS" if ok else "FAILED", "=====")
