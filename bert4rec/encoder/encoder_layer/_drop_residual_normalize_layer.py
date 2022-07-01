import pytorch_lightning as pl
import torch
import torch.nn as nn
from pydantic import BaseModel


class ConfDropResidualNormalizeLayer(BaseModel):
    """ConfDropResidaulNormalizeLayer class

    Args:
        dropout_rate (float): Dropout probability.
        norm_shape (int): Shape of the normalization layer.
        norm_eps (float): Epsilon for the normalization layer.
        name (str): Name of the layer.
    """
    dropout_rate: float
    norm_shape: int
    norm_eps: float
    name: str

class DropResidualNormalizeLayer(pl.LightningModule):
    """DropResidualNormalizeLayer _summary_

    Do following sequential steps:
        1. Dropout
        2. Add residual
        3. Layer normalization

    Args:
        c (ConfDropResidaulNormalizeLayer): The configuration of the layer.
    """
    def __init__(self, c: ConfDropResidualNormalizeLayer):
        super().__init__()
        self.c = c
        self.dropout = nn.Dropout(
            p=c.dropout_rate,
            inplace=True,
        )
        self.layer_norm = nn.LayerNorm(
            normalized_shape=c.norm_shape,
            eps=c.norm_eps,
        )

        nn.init.constant_(self.layer_norm.weight, 1.0)
        nn.init.constant_(self.layer_norm.bias, 0.0)
    
    def forward(
        self,
        batch_x: torch.tensor,
        identity: torch.tensor=None
    ):
        if self.c.dropout_rate:
            out = self.dropout(batch_x)
        
        if identity is not None:
            out += identity

        out = self.layer_norm(out)

        return out
