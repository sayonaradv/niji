from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer

TensorDict = dict[str, Tensor]

TextInput = str | list[str]
Batch = dict[str, str | Tensor]
PredResult = dict[str, Tensor | TensorDict]

ModelTokenizerPair = tuple[PreTrainedModel, PreTrainedTokenizer]
