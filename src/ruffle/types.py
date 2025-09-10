from torch import Tensor

BATCH = dict[str, str | Tensor]
MODEL_OUTPUT = tuple[Tensor, Tensor | None]
