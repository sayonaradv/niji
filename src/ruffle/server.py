from fastapi import Request
from litserve import LitAPI, LitServer
from torch import Tensor

from ruffle.core import Ruffle


class RuffleAPI(LitAPI):
    def setup(self, device: str) -> None:
        ckpt_path = "lightning_logs/version_2/checkpoints/epoch=07-val_loss=0.0454.ckpt"
        self.ruffle = Ruffle(ckpt_path=ckpt_path)

    def decode_request(self, request: Request):
        return request["text"]

    def predict(self, text: str) -> Tensor:
        return self.ruffle.classify(text, pretty_print=False)

    def encode_respose(self, probabilities: Tensor):
        return probabilities


if __name__ == "__main__":
    server = LitServer(RuffleAPI())
    server.run(port=8000)
