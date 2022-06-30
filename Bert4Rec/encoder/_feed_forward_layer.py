import pytorch_lightning as pl
import torch
import torch.nn as nn
from pydantic import BaseModel


class ConfFeedForwardLayer(BaseModel):
    """ConfFeedForwardLayer class

    Provides the configuration for the feed forward layer.

    Args:
        d_in (int): The dimension of the input.
        d_hid (int): The dimension of the hidden layer.
        activation (torch.Modlue): The activation function.
        name (str): The name of the layer.
    """
    d_in: int
    d_hid: int
    activation: torch.Modlue
    name: str

class FeedForwardLayer(pl.LightningModule):
    """FeedForwardLayer class

    Args:
        c (ConfFeedForwardLayer): The configuration for the feed forward layer.
    """
    def __init__(self, c: ConfFeedForwardLayer) -> None:
        super().__init__()

        self.fc1 = nn.Linear(
            in_features=c.d_in,
            out_features=c.d_hid,
        )

        self.act = c.activation

        self.fc2 = nn.Linear(
            in_features=c.d_mid,
            out_features=c.d_in,
        )

    def forward(self, batch_x: torch.tensor):
        hidden = self.fc1(batch_x)
        hidden = self.activation(hidden)

        return self.fc2(hidden)
