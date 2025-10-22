import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from niji.inference import TEST_CKPT_PATH, load_checkpoint

app = FastAPI(title="Niji Inference API")

model = load_checkpoint(ckpt_path=TEST_CKPT_PATH)
model.eval()
labels = model.hparams["label_names"]


class PredictRequest(BaseModel):
    input: str | list[str]


class PredictResponse(BaseModel):
    toxic: list[float]
    severe_toxic: list[float]
    obscene: list[float]
    threat: list[float]
    insult: list[float]
    identity_hate: list[float]


def classify_text(inputs: list[str]):
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.sigmoid(logits).detach().cpu()

    return {labels[i]: probs[:, i].tolist() for i in range(len(labels))}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    inputs = request.input if isinstance(request.input, list) else [request.input]
    results = classify_text(inputs)
    return results


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
