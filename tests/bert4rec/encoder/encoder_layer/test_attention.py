import pytest
import torch
from bert4rec.encoder.encoder_layer._attention import (ConfMultiHeadAttention,
                                                       MultiHeadAttention)


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

def test_multihead_attention_class(conf_multi_head_attention: ConfMultiHeadAttention):
    c: ConfMultiHeadAttention = conf_multi_head_attention
    attn: MultiHeadAttention = MultiHeadAttention(c)

    assert attn.Wk.in_features == c.d_model
    assert attn.Wq.in_features == c.d_model
    assert attn.Wv.in_features == c.d_model

    # batch size, sequence length
    B, S = 4, 128 

    # [B, S, dm]
    batch_x: torch.tensor = torch.randn(B, S, c.d_model)

    # [B, nh, S, S]
    attn_bias: torch.tensor = torch.randn(B, c.n_head, S, S)

    out = attn(
        batch_x=batch_x,
        attn_bias=attn_bias,
    )

    assert out.shape == (B, S, c.d_model)
