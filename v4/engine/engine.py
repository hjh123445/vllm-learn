# 兼容 shim：benchmark worker 用 from engine.engine import Engine 的路径约定
from engine.llm_engine import LLMEngine as Engine  # noqa: F401
