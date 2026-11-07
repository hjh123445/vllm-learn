"""Removable module ranges, owned by the worker's profiling window.

No tensor copies, synchronization, model-method replacement or model-side
profiling state. ATen and CUDA activities remain the profiler's responsibility.
"""
import re
import threading

from torch.profiler import record_function


class ModuleProfileRanges:
    """Label local decoder blocks, attention, MoE and router during capture.

    Hooks observe nn.Module calls only. Internal methods such as EP dispatch,
    compute_sorted and combine are not separate module boundaries.
    """

    def __init__(self, model, stage=0):
        self.model = model
        self.stage = stage
        self._handles = []

    def _attach(self, module, label):
        state = threading.local()

        def begin(_module, _args):
            context = record_function(label)
            context.__enter__()
            if not hasattr(state, "stack"):
                state.stack = []
            state.stack.append(context)

        def end(_module, _args, _output):
            stack = getattr(state, "stack", [])
            if stack:
                stack.pop().__exit__(None, None, None)

        self._handles.append(module.register_forward_pre_hook(begin))
        # Close the range even if this module or a nested child raises.
        self._handles.append(module.register_forward_hook(end, always_call=True))

    def install(self):
        if self._handles:
            return self
        try:
            for path, layer in self.model.named_modules():
                if path.endswith('vision_tower'):
                    self._attach(layer, f'stage{self.stage}.vision.encoder')
                vision = re.search(r'vision_tower\.blocks\.(\d+)$', path)
                if vision:
                    prefix = f'stage{self.stage}.vision.block{vision[1]}'
                    self._attach(layer, prefix)
                    self._attach(layer.attn, f'{prefix}.attention')
                    self._attach(layer.mlp, f'{prefix}.mlp')
                match = re.search(r"(?:^|\.)layers\.(\d+)$", path)
                if not match or not hasattr(layer, "self_attn") or not hasattr(layer, "mlp"):
                    continue  # Includes PPMissingLayer on non-local stages.
                prefix = f"stage{self.stage}.layer{match[1]}"
                self._attach(layer, prefix)
                self._attach(layer.self_attn, f"{prefix}.attention")
                self._attach(layer.mlp, f"{prefix}.moe")
                if hasattr(layer.mlp, "router"):
                    self._attach(layer.mlp.router, f"{prefix}.moe.router")
        except BaseException:
            self.remove()
            raise
        return self

    def remove(self):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *_exc):
        self.remove()
