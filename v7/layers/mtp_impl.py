# v7/layers/mtp_impl.py
"""n-gram draft 的独立实现（从 v7.0 的 mtp.py 拆出——组件文件可共存）。"""


class NGramDrafter:
    """n-gram draft：在上下文里找与当前尾部匹配的 n-gram，取其后续当候选。"""

    def __init__(self, n: int = 2, k: int = 4):
        self.n, self.k = n, k

    def draft(self, tokens: list[int]) -> list[int]:
        """tokens: prompt + 已生成的完整序列 → k 个候选 token。"""
        if len(tokens) < self.n + 1:
            return []
        # 从后往前找最近的 [t_{-n}..t_{-1}] 匹配（不含当前位置自身）
        tail = tuple(tokens[-self.n:])
        best = None
        for i in range(len(tokens) - self.n - 1, -1, -1):
            if tuple(tokens[i:i + self.n]) == tail:
                best = i
                break
        if best is None:
            return []
        cand = tokens[best + self.n:best + self.n + self.k]
        return cand[:self.k]
