# from litserve import LitAPI
from stormy.module import SequenceClassificationModule
import torch

# class StormyLitAPI(LitAPI):
#     def setup(self, ckpt_path: str, device: str):
#         """
#         Load the tokenizer and model, and move the model to the specified device.
#         """
#         self.lit_module = SequenceClassificationModule.load_from_checkpoint(ckpt_path)
#         self.lit_module.to(device)
#         self.lit_module.eval()
#
#         params = self.lit_module.
#
#

device = "cpu"
ckpt_path = "lightning_logs/version_4/checkpoints/epoch=19-val_loss=0.3007.ckpt"
checkpoint = torch.load(ckpt_path, map_location=device)
print(checkpoint["hyper_parameters"])
state_dict = checkpoint["state_dict"]
max_token_len = checkpoint["datamodule_hyper_parameters"]["max_token_len"]

