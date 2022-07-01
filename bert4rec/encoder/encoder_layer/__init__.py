"""encoder_layer module

This module implements the encoder layer of the transformer.

It is composed of the :py:class:multi-head attention and the positionwise feedforward network.
"""
from ._attention import ConfMultiHeadAttention, MultiHeadAttention
from ._drop_residual_normalize_layer import (ConfDropResidualNormalizeLayer,
                                             DropResidualNormalizeLayer)
from ._encoder_layer import ConfEncoderLayer, EncoderLayer
from ._feed_forward_layer import ConfFeedForwardLayer, FeedForwardLayer
