"""Opt-in first-forward activation capture, excluded from performance runs."""
import os
from pathlib import Path
import torch


def install_activation_capture(model, directory, stage=0, *, reference=False):
    """Capture matching attention/MoE boundaries and router outputs once.

    Only first/last 16 rows are retained; GPU-to-CPU copies synchronize and
    this facility must be disabled for throughput/profiler measurements.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    captured = set()
    handles = []
    full_rows = os.getenv('VLLM_LEARN_ACTIVATIONS_FULL_ROWS', '0') == '1'
    selected = os.getenv('VLLM_LEARN_ACTIVATIONS_LAYERS', '')
    selected_layers = {int(v) for v in selected.split(',') if v} if selected else None
    selected_modules = os.getenv('VLLM_LEARN_ACTIVATIONS_MODULES', '')
    selected_modules = {v.strip() for v in selected_modules.split(',') if v.strip()} or None
    call_index = int(os.getenv('VLLM_LEARN_ACTIVATIONS_CALL_INDEX', '0'))
    call_counts = {}
    pending_inputs = {}
    capture_weights = {v.strip() for v in
                       os.getenv('VLLM_LEARN_ACTIVATIONS_WEIGHTS', '').split(',')
                       if v.strip()}

    def pack(value):
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.ndim >= 2:
                value = value.reshape(-1, value.shape[-1])
                if not full_rows and value.shape[0] > 32:
                    value = torch.cat((value[:16], value[-16:]), dim=0)
            return value.cpu().clone()
        if isinstance(value, (tuple, list)):
            return [pack(v) for v in value]
        return None

    def attach(module, label, category):
        if selected_modules is not None and category not in selected_modules:
            return
        def inputs_from(args, kwargs):
            return args if args else tuple(kwargs[k] for k in
                ('hidden_states', 'input') if k in kwargs)

        def pre_hook(_module, args, kwargs):
            current_call = call_counts.get(label, 0)
            call_counts[label] = current_call + 1
            if current_call == call_index and label not in captured:
                # Save before the module runs: decoder blocks mutate residual
                # tensors in-place, so reading hook arguments afterward is wrong.
                pending_inputs[label] = pack(inputs_from(args, kwargs))

        def hook(_module, args, kwargs, output):
            if label not in pending_inputs or label in captured:
                return
            captured.add(label)
            record = dict(label=label, input=pending_inputs.pop(label), output=pack(output))
            if label in capture_weights and hasattr(_module, 'weight'):
                record['weight'] = _module.weight.detach().cpu().clone()
            torch.save(record, directory/f'stage{stage}.{label}.pt')
        handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(module.register_forward_hook(hook, with_kwargs=True))

    base = model.model
    attach(base.embed_tokens, 'embedding', 'embedding')
    for index, layer in enumerate(base.layers):
        if selected_layers is not None and index not in selected_layers:
            continue
        if not hasattr(layer, 'self_attn'):
            continue
        attach(layer.self_attn, f'layer{index}.attention', 'attention')
        attach(layer.mlp, f'layer{index}.moe', 'moe')
        attach(layer, f'layer{index}.block', 'block')
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            attach(getattr(layer.self_attn, name), f'layer{index}.{name}', 'projection')
        attach(layer.mlp.gate if reference else layer.mlp.router,
               f'layer{index}.router', 'router')
    attach(base.norm, 'final_norm', 'norm')
    if hasattr(model, 'lm_head'):
        attach(model.lm_head, 'lm_head', 'head')
    # Keep handles alive so callers can explicitly remove them if needed.
    return handles
