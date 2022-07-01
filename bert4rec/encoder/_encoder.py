"""

Returns:
    [type]: [description]
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pydantic import BaseModel

from .encoder_layer import ConfEncoderLayer, EncoderLayer


class ConfEncoder(BaseModel):
    """ConfEncoder class

    Provides the configuration for the encoder.

    Args:
        n_layer (int): Number of layers in the encoder.
        n_head (int): Number of heads in the encoder.
        d_key (int): Dimension of the key vectors.
        d_val (int): Dimension of the value vectors.
        d_model (int): Dimension of the model vectors.
        d_inner_hid (int): Dimension of the inner vectors.
    """
    n_layer: int
    conf_encoder_layer: ConfEncoderLayer
    name: str


class Encoder(pl.LightningModule):
    """Encoder class for the transformer.

    Args:
        config (ConfEncoder): The configuration for the encoder.
    """
    def __init__(self, c: ConfEncoder):
        super().__init__()
        self.c: ConfEncoderLayer = c

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(c.conf_encoder_layer) for _ in range(c.n_layer)
        ])
    
    def forward(self, batch_x: torch.tensor, attn_bias: torch.tensor):
        enc_out: torch.tensor = None

        for enc in self.encoder_layers:
            enc_out = enc(batch_x, attn_bias)
            batch_x = enc_out

        return enc_out
