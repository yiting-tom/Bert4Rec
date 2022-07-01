import pytest
import torch
from bert4rec.encoder.encoder_layer._drop_residual_normalize_layer import (
    ConfDropResidualNormalizeLayer, DropResidualNormalizeLayer)


@pytest.fixture(scope='session')
def conf_drop_residual_normalize_layer() -> ConfDropResidualNormalizeLayer:
    return ConfDropResidualNormalizeLayer(
        dropout_rate=0.1,
        norm_shape=128,     # d_model
        norm_eps=1e-5,
        name='test_drop_residual_norm_layer',
    )
        

def test_drop_residual_normalize_layer_class(conf_drop_residual_normalize_layer: ConfDropResidualNormalizeLayer):
    c: ConfDropResidualNormalizeLayer = conf_drop_residual_normalize_layer

    drn = DropResidualNormalizeLayer(c)

    B, S = 2, 3
    batch_x = torch.randn(B, S, c.norm_shape)
    identity = torch.randn_like(batch_x)

    out = drn(
        batch_x=batch_x,
        identity=identity,
    )

    assert out.shape == (B, S, c.norm_shape)
