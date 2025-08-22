from blanket import Blanket

ckpt_path: str = "lightning_logs/version_0/checkpoints/epoch=06-val_loss=0.0487.ckpt"

text: list[str] = [
    "you're such a bitch!",
    "can we be friends?",
    "the weather is great.",
]


_ = Blanket(ckpt_path).predict(text)
