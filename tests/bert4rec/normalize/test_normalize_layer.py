import pytest
import torch
from bert4rec.normalize._normalize_layer import (ConfNormalizeLayer,
                                                 NormalizeLayer)


@pytest.fixture(scope='session')
def conf_normalize_layer() -> ConfNormalizeLayer:
    return ConfNormalizeLayer(
        norm_shape=768,
        norm_eps=1e-5,
        name='normalize_layer',
    )

def test_normalize_layer_class(conf_normalize_layer: ConfNormalizeLayer):
    normalize_layer = NormalizeLayer(c=conf_normalize_layer)

    batch_x = torch.randn(1, 768)
    out = normalize_layer(batch_x)

    assert out.shape == batch_x.shape
