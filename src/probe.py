import torch
import torch.nn as nn
import torch.nn.functional as F
from nnsight import LanguageModel
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
import wandb

@dataclass
class ProbeConfig:
    steps: int = 50
    layer: int = 24 # ?
    use_wandb: bool = True
    wandb_project: str = "user-models"
    model: LanguageModel = LanguageModel("openai/gpt-oss-20b", device_map='device')


device = "cuda" if torch.cuda.is_available() else "cpu"

test_model = LanguageModel("openai/gpt-oss-20b", device_map='device')
print(test_model.config)
d_model = test_model.config.hidden_size
print(f"d_model = {d_model}")

# super simple lin probe code
# "do the simplest thing first :O"
class LinearProbe(torch.nn.Module):
    def __init__(self, d_in, n_cls, bias=True):
        super().__init__()
        self.W = torch.nn.Linear(d_in, n_cls, bias=bias)
    def forward(self, x):
        return self.W(x)

class LinearProbeTrainer:
    def __init__(self, config):
        self.cfg = config
        self.optim = torch.optim.AdamW(params=self.probe.parameters(),lr=1e-3)


    def init_probe():


    def get_loss(self):
        #TODO implement


    def train(self):
        for i in range(self.cfg.steps):
            loss = get_loss()
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            if self.cfg.use_wandb:
                wandb.log(
                    dict(
                        loss=loss.item(), # more later
                    )
                )

        if self.cfg.use_wandb:
            wandb.finish()

trainer = LinearProbeTrainer(ProbeConfig)
trainer.train()
