import pytorch_lightning as pl
import torch
import torch.nn as nn
from pydantic import BaseModel


class ConfNormalizeDropLayer(BaseModel):
    """ConfNormalizeDropLayer class

    Provides configuration for the NormalizeDropLayer class.

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

class NormalizeDropLayer(pl.LightningModule):
    """NormalizeDropLayer class

    This class implements the normalization layer with dropout.

    It will do the following:
        1. Layer Norm
        2. Dropout

    Args:
        c (ConfNormalizeDropLayer): Configuration for the NormalizeDropLayer class.
    """
    def __init__(self, c: ConfNormalizeDropLayer):
        super().__init__()
        self.c = c

        self.dropout = nn.Dropout(
            p=c.dropout_rate,
            inplace=True,
        )

        self.layer_norm = nn.LayerNorm(
            normalized_shape=c.norm_shape,
            eps=1e-5,
        )

        # Init weight and bias.
        nn.init.constant_(self.layer_norm.weight, 1.0)
        nn.init.constant_(self.layer_norm.bias, 0.0)
    

    def forward(self, batch_x: torch.tensor) -> torch.tensor:
        # Layer Norm
        normalized = self.layer_norm(batch_x)

        # Dropout
        if self.c.dropout_rate:
            out = self.dropout(normalized)

        return out
