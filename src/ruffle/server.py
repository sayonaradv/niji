import torch
from fastapi import Request
from litserve import LitAPI, LitServer

from ruffle.model import RuffleModel
from ruffle.predictor import AVAILABLE_MODELS


class SimpleLitAPI(LitAPI):
    def setup(self, device: str) -> None:
        model_url = AVAILABLE_MODELS["bert-tiny"]
        self.model = RuffleModel.load_from_checkpoint(model_url, map_location="cpu")
        self.model.eval()

    def decode_request(self, request: Request):
        return request["text"]

    def predict(self, text):
        """
        Perform the inference
        """
        with torch.no_grad():
            logits = self.model(text)[0].detach()
        return logits

    def encode_respose(self, logits):
        probabilities = torch.nn.functional.sigmoid(logits)
        return probabilities
        # label_names = self.model.label_names
        # if label_names is not None:
        #     return dict(zip(label_names, probabilities, strict=True))
        # else:
        #     return {"probabilities": probabilities}
        #


if __name__ == "__main__":
    server = LitServer(SimpleLitAPI())
    server.run(port=8000)
