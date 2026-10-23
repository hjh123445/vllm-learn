import dataclasses


@dataclasses.dataclass
class SamplingParams:
    temperature: float = 0.6
    top_p: float = 0.9
    max_tokens: int = 256
    ignore_eos: bool = False
    seed: int | None = None      # 预留：v0 尚未接采样 seed

    def __post_init__(self):
        assert self.temperature >= 0, "temperature 不能为负"
        assert self.max_tokens > 0, "max_tokens 必须大于 0"
