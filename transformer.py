import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from utils import clones


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, src_embedding, target_embedding, data_generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder =  decoder
        self.src_embed = src_embedding
        self.tgt_embed = target_embedding
        self.generator = data_generator

    def forward(self, src, target, src_mask, tgt_mask):
        """
        Take in and process source and target masked embeddings
        :param src:
        :param target:
        :param src_mask:
        :param tgt_mask:
        :return:
        """

        return self.decode(self.encode(src, src_mask), src_mask, target, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# Define the layer normalization layer
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# Define the encoder module
class Encoder(nn.Module):

    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layer = layer
        self.n = n
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# Define a sublayer class which takes care of the residual connections
class SubLayer(nn.Module):

    def __init__(self, size, dropout):
        super(SubLayer, self).__init__()
        self.size = size
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(self.size)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# Define the layer for an encoder module
class EncoderLayer(nn.Module):

    def __init__(self, size, dropout, self_attn, feed_forward):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.dropout = dropout
        self.self_attn = self_attn
        self.sublayers = clones(SubLayer(size, dropout), 2)
        self.feed_forward = feed_forward

    def forward(self, source, source_mask):
        x = self.self_attn(source,source,source, source_mask)
        x = self.sublayers[0](x)
        x = self.feed_forward(x)
        x = self.sublayers[1](x)
        return x


# Define the decoder module
class Decoder(nn.Module):

    def __init__(self, layer, n):
        super(Decoder, self).__init__()
        self.layer = layer
        self.n = n
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask )
        return self.norm(x)


# Define a single decoder layer
class DecoderLayer(nn.Module):

    def __init__(self, size, dropout, self_attn, src_attn,
                 feed_forward, d_model, vocab):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.dropout = nn.Dropout(dropout)
        self.attn = self_attn
        self.src_attn = src_attn
        self.sub_layers = clones(SubLayer(size, dropout), 3)
        self.feed_forward  = feed_forward
        self.generator = Generator(d_model=d_model, vocab=vocab)

    def forward(self, x, memory, source_mask, target_mask):
        x = self.attn(x, x, x, target_mask)
        x = self.sub_layers[0](x)
        x = self.src_attn(x, memory, memory, source_mask)
        x = self.sub_layers[1](x)
        x = self.feed_forward(x)
        x = self.sub_layers[2](x)
        return x


# Define the position wise feed forward neural network
class PositionWiseFeedForwardNN(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PositionWiseFeedForwardNN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.dense_1 = nn.Linear(self.d_model, self.d_ff)
        self.dense_2 = nn.Linear(self.d_ff, self.d_model)

    def forward(self, x):
        x = self.dense_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        return x


"""
Modify the self attention sub layer in the decoder stack to prevent positions from attending to subsequent positions.
This masking combined with the fact that the output is offset by a position of 1 ensures that the prediction for position 
i can only depend on positions less than i.
"""


def subsequent_masking(size):
    """
    Mask out subsequent positions
    :param size:
    :return:
    """

    attn_shape = (1, size, size)
    # np.triu returns the upper triangular matrix with values below the kth diagonal set to 0
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0






















