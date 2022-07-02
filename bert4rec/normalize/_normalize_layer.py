import pytorch_lightning as pl
import torch
import torch.nn as nn
from pydantic import BaseModel


class ConfNormalizeLayer(BaseModel):
    norm_shape: int
    norm_eps: float
    name: str

class NormalizeLayer(pl.LightningModule):
    def __init__(self, c: ConfNormalizeLayer):
        super().__init__()
        self.c = c
        self.layer_norm = nn.LayerNorm(
            normalized_shape=c.norm_shape,
            eps=c.norm_eps,
        )
    
    def forward(self, batch_x: torch.tensor):
        return self.layer_norm(batch_x)
