import pytorch_lightning as pl
import torch
from pydantic import BaseModel

from ._attention import ConfMultiHeadAttention, MultiHeadAttention
from ._drop_residual_normalize_layer import (ConfDropResidualNormalizeLayer,
                                             DropResidualNormalizeLayer)
from ._feed_forward_layer import ConfFeedForwardLayer, FeedForwardLayer


class ConfEncoderLayer(BaseModel):
    """ConfEncoderLayer class

    Args:
        conf_multi_head_attn (ConfMultiHeadAttention): The configuration of the Multi head attention.
        conf_drop_residual_normalize_layer (ConfDropResidualNormalizeLayer): The configuration of the Drop residual normalize layer.
        conf_feed_forward_layer (ConfFeedForwardLayer): The configuration of the Feed forward layer.
        name (str): The name of the layer.
    """
    conf_multihead_attn: ConfMultiHeadAttention
    conf_drop_residual_norm_layer: ConfDropResidualNormalizeLayer
    conf_feed_forward_layer: ConfFeedForwardLayer
    name: str

class EncoderLayer(pl.LightningModule):
    """EncoderLayer class

    The encoder layer do the following sequential steps:
        1. Multi head attention
        2. Drop residual normalize layer (multi head attention + identity: batch_x)
        3. Feed forward layer
        4. Drop residual normalize layer (output of step 3 + identity: output of step 2)

    Args:
        c (ConfEncoderLayer): The configuration of the encoder layer.
    """
    def __init__(self, c: ConfEncoderLayer):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(
            c.conf_multihead_attn
        )
        self.drop_residual_norm_layer_1 = DropResidualNormalizeLayer(
            c.conf_drop_residual_norm_layer
        )
        self.positionwise_feed_layer = FeedForwardLayer(
            c.conf_feed_forward_layer
        )
        self.drop_residual_norm_layer_2 = DropResidualNormalizeLayer(
            c.conf_drop_residual_norm_layer
        )
    
    def forward(self, batch_x: torch.tensor, attn_bias: torch.tensor):
        multihead_attn_out = self.multihead_attn(
            batch_x=batch_x,
            attn_bias=attn_bias
        )
        residual_multihead_attn_out = self.drop_residual_norm_layer_1(
            batch_x=multihead_attn_out,
            identity=batch_x,
        )
        ff_out = self.positionwise_feed_layer(
            batch_x=residual_multihead_attn_out,
        )
        out = self.drop_residual_norm_layer_2(
            batch_x=ff_out,
            identity=residual_multihead_attn_out,
        )

        return out
