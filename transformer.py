import torch.nn as nn
import torch.nn.functional as F
import torch


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


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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














