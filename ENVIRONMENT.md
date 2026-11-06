# 环境与复现范围

当前维护目标：Linux、NVIDIA A800（SM80）、Qwen3-Omni-30B-A3B-Instruct文本Thinker。分析/合并trace只需Python标准库，可在Windows运行。多卡推理使用NCCL，不能在Windows本机等价复现。

本次2026-11-06 GPU工具验证与SiLU基准使用Python3.10、PyTorch2.9.1+cu130、A800-SXM4-80GB；其余精确包版本与GPU状态在每次capture的manifest里。不要用本文替代实验环境指纹。真实30B历史数据来自各自保存的deployment/queue，不保证与当前环境完全相同。

```bash
# 先按驱动/平台安装兼容的CUDA版PyTorch，再装本项目依赖。
python -m pip install -r requirements.txt
python -m pip install pytest
```

`requirements.txt`给出接口依赖下界，不承诺所有版本组合都已验证。HF完整Omni参考类和native grouped_mm需要额外确认相应Transformers/PyTorch版本；默认参考专家loop不要求native grouped_mm。可选FlashAttention需要与PyTorch/CUDA ABI匹配；本项目支持SDPA参考路径，不要求为采集工具强装FlashAttention。

vLLM建议使用独立环境，避免安装时替换共享环境的PyTorch。与自研引擎对照时明确记录两边版本、dtype、Attention/专家后端、eager/Graph模式及输入；软件栈不同可能解释部分性能差异，不能只说“同机”就认定条件完全一致。

```bash
nvidia-smi
nvidia-smi topo -m
python -c 'import torch,triton,transformers; print(torch.__version__,triton.__version__,transformers.__version__)'
```

多卡总数为DP×PP×TP（本工具不暴露CP轴），必须有对应数量的可见GPU。`CUDA_VISIBLE_DEVICES`映射的是物理卡列表，worker内部使用局部编号。不要一边只暴露GPU7，一边启动TP4；这已在采集入口加预检查。

CPU-only测试若系统Python加载了冲突的OpenMP runtime，请使用干净虚拟环境。不要设置绕过冲突的环境开关后把异常环境当作正式性能数据。
