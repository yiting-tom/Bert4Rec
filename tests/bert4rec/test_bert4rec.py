import pytest
import torch

from bert4rec.bert4rec import Bert4Rec, ConfBert4Rec


@pytest.mark.order(8)
def test_bert4rec_class(conf_bert4rec: ConfBert4Rec):
    c = conf_bert4rec
    bert4rec = Bert4Rec(c=c)

    src_id = torch.arange(
        start=0,
        end=8,
        dtype=torch.long,
    )
    position_ids = torch.arange(
        start=0,
        end=8,
        dtype=torch.long,
    )
    sent_ids = torch.randint(
        low=0,
        high=conf_bert4rec.sent_types,
        size=(1, 8),
    )
    input_mask = torch.ones(
        size=(1, 8),
    )
    mask_pos = torch.randint(
        low=0,
        high=8,
        size=(1, c.emb_size),
    )

    out = bert4rec(
        src_ids=src_id,
        position_ids=position_ids,
        sent_ids=sent_ids,
        input_mask=input_mask,
        mask_pos=mask_pos,
    )

    assert out.shape[-1] == conf_bert4rec.voc_size
