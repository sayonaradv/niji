from torch import Tensor

BATCH = dict[str, str | Tensor]
MODEL_OUTPUT = Tensor | tuple[Tensor, Tensor]
