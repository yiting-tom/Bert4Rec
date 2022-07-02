import pytest
import torch
from bert4rec.normalize._normalize_dropout_layer import (
    ConfNormalizeDropLayer, NormalizeDropLayer)


@pytest.mark.order(6)
def test_normalize_drop_layer_class(conf_normalize_drop_layer: ConfNormalizeDropLayer):
    normalize_drop_layer = NormalizeDropLayer(c=conf_normalize_drop_layer)

    batch_x = torch.randn(1, conf_normalize_drop_layer.norm_shape)
    out = normalize_drop_layer(batch_x)

    assert out.shape == batch_x.shape
