import pytest
import torch
import torch.nn as nn

from bert4rec.encoder import (ConfDropResidualNormalizeLayer, ConfEncoder,
                              ConfEncoderLayer, ConfFeedForwardLayer,
                              ConfMultiHeadAttention)


@pytest.fixture(scope='session')
def conf_multi_head_attention() -> ConfMultiHeadAttention:
    return ConfMultiHeadAttention(
        n_head=8,
        d_key=16,
        d_val=16,
        d_model=128,
        dropout_rate=0.1,
        name='test_multihead_attention'
    )


@pytest.fixture(scope='session')
def conf_feed_forward_layer() -> ConfFeedForwardLayer:
    return ConfFeedForwardLayer(
        d_in=128,       # d_model
        d_hid=256,
        activation=nn.ReLU(inplace=True),
        name='test_feed_forward_layer',
    )

@pytest.fixture(scope='session')
def conf_drop_residual_normalize_layer() -> ConfDropResidualNormalizeLayer:
    return ConfDropResidualNormalizeLayer(
        dropout_rate=0.1,
        norm_shape=128,     # d_model
        norm_eps=1e-5,
        name='test_drop_residual_norm_layer',
    )

@pytest.fixture(scope='session')
def conf_encoder_layer(
    conf_multi_head_attention,
    conf_drop_residual_normalize_layer,
    conf_feed_forward_layer,
) -> ConfEncoderLayer:

    return ConfEncoderLayer(
        conf_multihead_attn=conf_multi_head_attention,
        conf_drop_residual_norm_layer=conf_drop_residual_normalize_layer,
        conf_feed_forward_layer=conf_feed_forward_layer,
        name='test_encoder_layer',
    )

@pytest.fixture(scope='session')
def conf_encoder(conf_encoder_layer: ConfEncoderLayer) -> ConfEncoder:
    return ConfEncoder(
        n_layer=4,
        conf_encoder_layer=conf_encoder_layer,
        name='test_encoder',
    )

from bert4rec.normalize import ConfNormalizeDropLayer, ConfNormalizeLayer


@pytest.fixture(scope='session')
def conf_normalize_drop_layer() -> ConfNormalizeDropLayer:
    return ConfNormalizeDropLayer(
        dropout_rate=0.1,
        norm_shape=128,
        norm_eps=1e-5,
        name='normalize_drop_layer',
    )

@pytest.fixture(scope='session')
def conf_normalize_layer() -> ConfNormalizeLayer:
    return ConfNormalizeLayer(
        norm_shape=128,
        norm_eps=1e-5,
        name='normalize_layer',
    )

from bert4rec import ConfBert4Rec


@pytest.fixture(scope='session')
def conf_bert4rec(
    conf_normalize_drop_layer,
    conf_normalize_layer,
    conf_encoder
) -> ConfBert4Rec:

    return ConfBert4Rec(
        device=torch.device('cpu'),
        dtype=torch.float32,
        emb_size=128,
        voc_size=54546,
        sent_types=2,
        max_pos_seq_len=50,
        initializer_range=0.02,
        hidden_act=nn.GELU(),
        conf_normalize_drop_layer=conf_normalize_drop_layer,
        conf_normalize_layer=conf_normalize_layer,
        conf_encoder=conf_encoder,
    )
