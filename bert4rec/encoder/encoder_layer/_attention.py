import pytorch_lightning as pl
import torch
import torch.nn as nn
from pydantic import BaseModel


class ConfMultiHeadAttention(BaseModel):
    """ConfMultiHeadAttention class

    Args:
        n_head (int): The number of attention heads.
        d_key (int): The dimension of the key.
        d_val (int): The dimension of the value.
        d_model (int): The dimension of the model.
        dropout_rate (float): The dropout rate, default is 0.0.
        name (str): The name of the layer, default is 'multi_head_attn'.
    """
    n_head: int
    d_key: int
    d_val: int
    d_model: int
    dropout_rate: float
    name: str

    @classmethod
    def new_default(cls):
        return cls(
            n_head=8,
            d_key=64,
            d_val=64,
            d_model=512,
            dropout_rate=0.1,
            name='multi_head_attn',
        )

class MultiHeadAttention(pl.LightningModule):
    """MultiHeadAttention _summary_

    Note:
        `B`: batch size
        `S`: sequence length
        `dm`: d_model (dimension of model)
        `dh`: d_head (dimension of head)
        `nh`: n_head (number of heads) 

    Warning:
        Here dm = dh * nh

    Args:
        c (ConfMultiHeadAttention): The configuration of the Multi head attention.
    """
    def __init__(self, c: ConfMultiHeadAttention):
        super().__init__()
        self.c = c
        self.Wq = nn.Linear(
            in_features=c.d_model,
            out_features=c.d_key * c.n_head,
        )
        self.Wk = nn.Linear(
            in_features=c.d_model,
            out_features=c.d_key * c.n_head,
        )
        self.Wv = nn.Linear(
            in_features=c.d_model,
            out_features=c.d_val * c.n_head,
        )
        self.out = nn.Linear(
            in_features=c.d_key * c.n_head,
            out_features=c.d_model,
        )
        self.dropout = nn.Dropout(
            p=c.dropout_rate,
            inplace=True,
        )
    
    def forward(
        self,
        batch_x: torch.tensor,
        attn_bias: torch.tensor
    ):
        q: torch.tensor = self.Wq(batch_x)
        k: torch.tensor = self.Wk(batch_x)
        v: torch.tensor = self.Wv(batch_x)

        # q = [B, S, dm]
        B, S, dm = q.size()

        # get the dh (dimension of head) from d_model.
        dh = dm // self.c.n_head

        # to_shape = [B, -1, nh, dh]
        to_shape = [B, -1, self.c.n_head, dh]

        # q: [B, S, d * n] -> [B, S, nh, dh]
        q = q.view(to_shape) 
        # q: [B, S, nh, dh] -> [B, nh, S, dh]
        # q = torch.einsum("BSnd -> BnSd", q) the einsum is slow
        q = q.transpose(1, 2)

        # k: [B, S, d * n] -> [B, S, nh, dh]
        k = k.view(to_shape)
        # k: [B, S, nh, dh] -> [B, nh, dh, S]
        # k = torch.einsum("BSnd -> BndS", k)
        k = k.transpose(1, 2).transpose(2, 3)

        # v: [B, S, d * n] -> [B, S, nh, dh]
        v = v.view(to_shape)
        # v: [B, S, nh, dh] -> [B, nh, S, dh]
        # v = torch.einsum("BSnd -> BnSd", v)
        v = v.transpose(1, 2)

        # attn_bias: [B, nh, S, S]
        attn_score = (q @ k) / torch.sqrt(torch.tensor(dh, dtype=q.dtype))

        # adding attention bias
        if attn_bias is not None:
            attn_score += attn_bias
        
        # softmax(QK)
        h = torch.softmax(input=attn_score, dim=-1)

        # Dropout
        if self.c.dropout_rate:
            h = self.dropout(h)

        # [B, nh, S, dh] @ [B, nh, S, dv] -> [B, S, dm]
        out = h @ v

        # h: [B, nh, S, dh] -> [B, S, dm]
        out = out.transpose(1, 2).reshape(B, S, dm)

        # out: [B, S, dm]
        return self.out(out)
