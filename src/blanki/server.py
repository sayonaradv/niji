import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from blanki.inference import TEST_CKPT_PATH, load_checkpoint

app = FastAPI()

model = load_checkpoint(ckpt_path=TEST_CKPT_PATH)
model.eval()

# Get labels from model checkpoint
labels = model.hparams["label_names"]


class PredictRequest(BaseModel):
    input: str | list[str]


@app.post("/predict")
def predict(request: PredictRequest):
    with torch.no_grad():
        logits = model(request.input).detach().cpu()
        probabilities = torch.sigmoid(logits)

    # Map labels to probabilities
    result = dict(zip(labels, probabilities, strict=True))

    return result


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
