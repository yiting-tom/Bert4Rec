import pytest
import torch
from bert4rec.normalize._normalize_layer import (ConfNormalizeLayer,
                                                 NormalizeLayer)


@pytest.mark.order(7)
def test_normalize_layer_class(conf_normalize_layer: ConfNormalizeLayer):
    normalize_layer = NormalizeLayer(c=conf_normalize_layer)

    batch_x = torch.randn(1, conf_normalize_layer.norm_shape)
    out = normalize_layer(batch_x)

    assert out.shape == batch_x.shape
