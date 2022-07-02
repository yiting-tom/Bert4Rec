import pytest
import torch
from bert4rec.normalize._normalize_dropout_layer import (
    ConfNormalizeDropLayer, NormalizeDropLayer)


@pytest.fixture(scope='session')
def conf_normalize_drop_layer() -> ConfNormalizeDropLayer:
    return ConfNormalizeDropLayer(
        dropout_rate=0.1,
        norm_shape=768,
        norm_eps=1e-5,
        name='normalize_drop_layer',
    )


def test_normalize_drop_layer_class(conf_normalize_drop_layer: ConfNormalizeDropLayer):
    normalize_drop_layer = NormalizeDropLayer(c=conf_normalize_drop_layer)

    batch_x = torch.randn(1, 768)
    out = normalize_drop_layer(batch_x)

    assert out.shape == batch_x.shape
