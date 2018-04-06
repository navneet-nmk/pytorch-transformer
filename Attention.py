"""
This File contains the attention mechanism as defined in the paper

*********
ATTENTION
*********

An attention function can be described as mapping a query and a set of key-value pairs to an output,
where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values,
where the weight assigned to each value is computed by a compatibility function of the query with the
corresponding key.

"""

import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from utils import clones


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute scaled do product attention
    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :return:
    """
    dim_k = query.size(-1)
    scores  = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


"""
Multi-head attention allows the model to jointly attend to information from 
different representation subspaces at different positions. With a single attention head, 
averaging inhibits this. 
"""


class MultiHeadAttention(nn.Module):

    def __init__(self, num_attention_layers, dim_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.h = num_attention_layers
        self.dropout = dropout
        self.d_model = dim_model
        self.d_k = dim_model//self.h
        self.linears = clones(nn.Linear(dim_model, dim_model), 4)
        self.attn = None
        self.drop = nn.Dropout(self.dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Do all the linear projections in batch from d_model to h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)





