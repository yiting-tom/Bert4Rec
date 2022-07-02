import pytest
import torch
from bert4rec.encoder.encoder_layer._drop_residual_normalize_layer import (
    ConfDropResidualNormalizeLayer, DropResidualNormalizeLayer)


@pytest.mark.order(3)
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
