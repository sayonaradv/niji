import torch
from fastapi import Request
from litserve import LitAPI, LitServer

from blanki import inference


class BlankiAPI(LitAPI):
    def setup(self, device: str) -> None:
        ckpt_path: str = inference.TEST_CKPT_PATH
        self.model = inference.load_checkpoint(ckpt_path=ckpt_path)
        self.model.eval()

    def decode_request(self, request: Request) -> str | list[str]:
        return request["text"]

    def predict(self, x: str | list[str]) -> torch.Tensor:
        with torch.no_grad():
            output: torch.Tensor = self.model(x)
        return output

    def encode_response(
        self, output: torch.Tensor
    ) -> list[float] | dict | list[dict[str, float]]:
        probabilities: torch.Tensor = torch.sigmoid(output).detach().cpu()

        if self.model.hparams["label_names"] is not None:
            # Handle batch inputs (multiple texts)
            if probabilities.dim() > 1:
                # Convert 2D tensor to list of dictionaries
                result: list[dict[str, float]] = []
                for i in range(probabilities.shape[0]):
                    result.append(
                        dict(
                            zip(
                                self.model.hparams["label_names"],
                                probabilities[i].tolist(),
                                strict=True,
                            )
                        )
                    )
                return result
            else:
                # Single input
                return dict(
                    zip(
                        self.model.hparams["label_names"],
                        probabilities.tolist(),
                        strict=True,
                    )
                )
        else:
            # Return raw probabilities
            return probabilities.tolist()


if __name__ == "__main__":
    server = LitServer(BlankiAPI())
    server.run(port=8000)
