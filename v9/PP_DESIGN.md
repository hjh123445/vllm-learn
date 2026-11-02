# v9 Pipeline Parallelism

v9 keeps one scheduler in `EngineCore`, as vLLM v1 does. A scheduled batch is
turned into immutable `PipelinePlan` messages and split into ordered
microbatches. Every plan carries a `microbatch_id` and `virtual_engine` slot;
responses are matched by ID before the scheduler retires the corresponding
sequences.

## Configuration

```python
EngineCore(
    model_path,
    pp_size=2,
    config_kwargs={"pipeline_microbatches": 4},
    gpus=[0, 1],
)
```

`virtual_engine_size` is an alias for `pipeline_microbatches`. PP=1 always
uses one plan and keeps the single-card path unchanged. With PP>1 and
`pipeline_microbatches=1`, behavior is equivalent to the old lockstep path.
Increasing the value sends several plans before collecting the last-stage
responses, filling the forward pipeline.

For a forward-only analogue of DeepSeek's zero-bubble idea, set
`pipeline_schedule="zero_bubble"`. The engine keeps at least two pipeline
depths worth of microbatch slots in flight (bounded by the number of scheduled
sequences), while stage-to-stage sends remain asynchronous. This hides the
steady-state wait between stages, but it is not the training ZB-H1/ZB-V
schedule: there is no backward pass and a batch with fewer than `2 * pp_size`
sequences still has fill/drain bubbles.

## Transport invariants

- Each PP lane is formed by ranks with the same TP/EP/DCP coordinate.
- The model owns no rank arithmetic or raw `dist.send/recv`;
  `PipelineTransport` owns the shape header and activation protocol.
- Intermediate activations use nonblocking sends. Send buffers stay referenced
  until their `Work` completes; workers drain all pending sends before NCCL
  teardown.
- Only stage `pp_size - 1`, inner rank zero returns sampled token IDs. For
  TP/EP without PP, only global rank zero returns them, so response pipes do
  not accumulate unused outputs.
- A microbatch is retired exactly once. Sequence mutation and KV-cache block
  release remain in the EngineCore scheduler.

The current scheduler partitions complete scheduled sequences. This preserves
KV-cache ownership and request ordering. A single very long sequence therefore
cannot be split across microbatches; use chunked prefill or a larger request
batch to expose more pipeline parallelism.

## Validation

`tests/test_pipeline_schedule.py` checks ordering, ownership, and message
identity without loading a model. Use an EngineCore smoke script with
`pp_size=2` and compare `pipeline_microbatches=1` and `4` with identical
prompts. Verify token equality before comparing TTFT, TPOT, and throughput.
The older `tests/run_pp.py` is a legacy standalone stage test and does not
exercise EngineCore microbatching.

