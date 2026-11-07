"""mrope 位置构造器对拍:闭式 mrope_positions vs HF get_rope_index 复演(仅图像)。

HF 逻辑(transformers qwen 系列 get_rope_index 的图像分支简版):
  text: 三轴同值,空间坐标从 st 起逐 token +1
  image (llm grid t,gh,gw): t 轴 = 0..t-1(每帧常数), h/w 轴 = 网格 raster,
                            整体加 st;st 更新 = 该段三轴 max + 1
  rope delta = max(pos)+1 - seq_len
本测试用同一套随机用例同时跑两份实现并比较逐位一致。
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layers.mm_preprocess import mrope_positions

IMG_TOKEN = 151655  # 占位即可,两份实现只看语义不看具体 id


def hf_reference_positions(prompt_len, images):
    """HF get_rope_index 图像语义的独立复写(参照实现)。"""
    pos3 = np.zeros((3, prompt_len), dtype=np.int64)
    st = 0
    cursor = 0
    for pos, t, gh, gw in sorted(images, key=lambda x: x[0]):
        text_n = pos - cursor
        for k in range(text_n):
            pos3[:, cursor + k] = st + k
        st += text_n
        cursor = pos
        llm = t * gh * gw
        t_index = np.arange(t).reshape(-1, 1).repeat(gh * gw, 1).reshape(-1)
        h_index = np.arange(gh).reshape(1, -1, 1).repeat(t, 0).repeat(gw, 2).reshape(-1)
        w_index = np.arange(gw).reshape(1, 1, -1).repeat(t, 0).repeat(gh, 1).reshape(-1)
        seg = np.stack([t_index, h_index, w_index], 0) + st
        pos3[:, pos:pos + llm] = seg
        st = int(seg.max()) + 1
        cursor = pos + llm
    tail = prompt_len - cursor
    for k in range(tail):
        pos3[:, cursor + k] = st + k
    st += tail
    return pos3, st - prompt_len


def _case(rng, n_images):
    images = []
    cur = 3
    for _ in range(n_images):
        pos = cur + int(rng.integers(0, 5))
        t = int(rng.choice([1, 1, 2]))
        gh = int(rng.integers(1, 8)) * 2
        gw = int(rng.integers(1, 8)) * 2
        images.append((pos, t, gh // 2, gw // 2))
        cur = pos + t * (gh // 2) * (gw // 2)
    prompt_len = cur + int(rng.integers(1, 6))
    return prompt_len, images


def test_mrope_positions_match_hf():
    rng = np.random.default_rng(7)
    for trial in range(200):
        prompt_len, images = _case(rng, int(rng.integers(1, 4)))
        got_pos3, got_delta = mrope_positions(prompt_len, images)
        ref_pos3, ref_delta = hf_reference_positions(prompt_len, images)
        assert got_delta == ref_delta, (trial, got_delta, ref_delta)
        assert np.array_equal(got_pos3, ref_pos3), (trial, images)
    # 纯文本退化:pos3 恒等 arange,delta=0
    p3, d = mrope_positions(37, [])
    assert d == 0 and np.array_equal(p3, np.stack([np.arange(37)] * 3))
    print('test_mrope_positions OK (200 random cases + text-only)')


if __name__ == '__main__':
    main()
