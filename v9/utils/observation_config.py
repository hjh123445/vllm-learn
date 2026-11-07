"""Experiment entry-point configuration; not imported by the engine."""
import os
from pathlib import Path


def configure_observation(output, *, profile=False, start_message=1, messages=0,
                          record_shapes=False):
    """Keep inherited diagnostic settings out of throughput measurements."""
    if start_message < 1 or messages < 0:
        raise ValueError("profile start_message must be positive; messages must be >= 0 (0 = until shutdown)")
    for key in tuple(os.environ):
        if key.startswith(("VLLM_LEARN_PROFILE_", "VLLM_LEARN_ACTIVATIONS_", "VLLM_LEARN_LOGITS_", "VLLM_LEARN_SDPA_CAPTURE_")):
            os.environ.pop(key)
    for key in ("VLLM_LEARN_CUDA_PROFILE", "VLLM_LEARN_TRACE_DIR", "VLLM_LOGIT_DUMP"):
        os.environ.pop(key, None)
    directory = Path(output).resolve()
    os.environ["VLLM_LEARN_METRICS_DIR"] = str(directory)
    if profile:
        trace = directory / "trace"
        if trace.exists() and any(trace.iterdir()):
            raise ValueError(f"profile trace directory must be empty: {trace}; use a new output directory")
        trace.mkdir(parents=True, exist_ok=True)
        os.environ.update(VLLM_LEARN_CUDA_PROFILE="1", VLLM_LEARN_TRACE_DIR=str(directory / "trace"),
            VLLM_LEARN_PROFILE_MODE="window", VLLM_LEARN_PROFILE_START_MESSAGE=str(start_message),
            VLLM_LEARN_PROFILE_MESSAGES=str(messages),
            VLLM_LEARN_PROFILE_RECORD_SHAPES=str(int(record_shapes)))
