#%%
from typing import Any, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pydantic import BaseModel

from bert4rec.normalize import (ConfNormalizeDropLayer, ConfNormalizeLayer,
                                NormalizeDropLayer, NormalizeLayer)

from .encoder import ConfEncoder, Encoder


class ConfBert4Rec(BaseModel):
    device: Any
    dtype: Any
    emb_size: int
    voc_size: int
    sent_types: int
    max_pos_seq_len: int
    hidden_act: Any
    initializer_range: float
    conf_normalize_drop_layer: ConfNormalizeDropLayer
    conf_normalize_layer: ConfNormalizeLayer
    conf_encoder: ConfEncoder

    @property
    def n_head(self):
        return self.conf_encoder\
            .conf_encoder_layer\
            .conf_multihead_attn\
            .n_head

#%%
class Bert4Rec(pl.LightningModule):
    """Bert4Rec class

    Args:
        c (ConfBert4Rec): The configuration for the Bert4Rec model.
    """
    def __init__(self, c: ConfBert4Rec):
        super().__init__()
        self.c: ConfBert4Rec = c

        # Initialize the embedding layers.
        self.word_emb = nn.Embedding(
            num_embeddings=c.voc_size,
            embedding_dim=c.emb_size,
            sparse=False,
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=c.max_pos_seq_len,
            embedding_dim=c.emb_size,
            sparse=False,
        )
        self.sent_emb = nn.Embedding(
            num_embeddings=c.sent_types,
            embedding_dim=c.emb_size,
            sparse=False,
        )
        self.__init_truncatedNormal_embeddings(
            layers=[self.word_emb, self.pos_emb, self.sent_emb],
            mean=0.0,
            std=c.initializer_range,
        )
    
        # Initialize the Layer Norm Layer.
        self.norm_drop_before_encoder_layer = NormalizeDropLayer(
            c=c.conf_normalize_drop_layer
        )

        # Initialize the encoder.
        self.encoder = Encoder(
            c=c.conf_encoder
        )

        self.mask_trans_feat = nn.Linear(
            in_features=c.emb_size,
            out_features=c.emb_size,
        )

        self.mask_trans_activation = c.hidden_act

        self.mask_post_process_layer = NormalizeLayer(
            c=c.conf_normalize_layer
        )

        self.mask_lm_out_bias = torch.nn.Parameter(
            data=torch.zeros(
                size=[self.c.voc_size],
                dtype=self.c.dtype,
                device=self.c.device,
            ),
            requires_grad=True,
        )
    
    def forward(
        self,
        src_ids: torch.tensor,
        position_ids: torch.tensor,
        sent_ids: torch.tensor,
        input_mask: torch.tensor,
        mask_pos: torch.tensor,
    ):
        # Embed the source ids.
        word_emb_out = self.word_emb(src_ids)
        pos_emb_out = self.pos_emb(position_ids)
        sent_emb_out = self.sent_emb(sent_ids)

        # Sum the embeddings.
        emb_out = word_emb_out + pos_emb_out + sent_emb_out

        # Apply the Layer Norm Layer.
        emb_out = self.norm_drop_before_encoder_layer(emb_out)

        # Initialize the self attention mask.
        self_attn_mask = input_mask @ input_mask.T
        self_attn_mask = 1e4 * (self_attn_mask - 1)
        n_head_self_attn_mask = torch.stack(
            tensors=[self_attn_mask] * self.c.n_head,
            dim=1,
        ).requires_grad_(False)
        
        # Mask the embedding by `n_head_self_attn_mask`
        # for masking the backward attention.
        self.enc_out = self.encoder(
            batch_x=emb_out,
            attn_bias=n_head_self_attn_mask,
        )

        # Reshape the output of encoder to
        # [batch_size*seq_len, emb_size]
        # for the position masking process.
        reshaped_emb_out = torch.reshape(
            input=self.enc_out,
            shape=[-1, self.c.emb_size],
        )

        # The position masking process.
        # Mask the embedding by `mask_pos` along the sequence dimension.
        mask_feat = torch.gather(
            input=reshaped_emb_out,
            index=mask_pos,
            dim=0,
        )

        mask_trans_feat_out = self.mask_trans_feat(mask_feat)
        mask_trans_feat_out = self.mask_trans_activation(mask_trans_feat_out)
        mask_trans_feat_out = self.mask_post_process_layer(mask_trans_feat_out)

        for name, param in self.named_parameters():
            if name == "word_emb.weight":
                y_tensor = param
                break

        ff_out = mask_trans_feat_out @ y_tensor.T
        ff_out += self.mask_lm_out_bias

        return ff_out

    def __init_truncatedNormal_embeddings(
        self,
        layers: List[nn.Embedding],
        mean: float=0.0,
        std: float=1.0,
    ):
        """__init_truncatedNormal_embeddings method

        Provides a truncated normal initialization for the embedding layers.

        Args:
            layers (List[nn.Embedding]): The embedding layers.
            mean (float, optional): The mean. Defaults to 0.0.
            std (float, optional): The std. Defaults to 1.0.
        """
        for layer in layers:
            nn.init.trunc_normal_(
                tensor=layer.weight,
                mean=mean,
                std=std,
            )

