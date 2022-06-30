from turtle import forward

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pydantic import BaseModel


class ConfDropResidaulNormalizeLayer(BaseModel):
    """ConfDropResidaulNormalizeLayer class

    Process:
        1. Dropout
        2. Residual addition
        3. Normalize

    Args:
        dropout_rate (float): Dropout probability.
        norm_shape (int): Shape of the normalization layer.
        norm_eps (float): Epsilon for the normalization layer.
        name (str): Name of the layer.
    """
    dropout_rate: float = 0.0
    norm_shape: int = 768
    norm_eps: float = 1e-5
    name: str = ''

class DropResidualNormalizeLayer(pl.LightningModule):
    """DropResidualNormalizeLayer _summary_

    Args:
        c (ConfDropResidaulNormalizeLayer): The configuration of the layer.
    """
    def __init__(self, c: ConfDropResidaulNormalizeLayer):
        super().__init__()
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
        out: torch.tensor,
        prev_out: torch.tensor=None
    ):
        if self.dropout_rate:
            out = self.dropout(out)
        
        if prev_out is not None:
            out = out + prev_out

        out = self.layer_norm(out)

        return out
