import torch.nn
from torch import nn

from Modules.TransformerEncoderLayer import TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_heads,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 dropout=None, scaled=True, ff_size=-1,
                 max_seq_len=-1, ff_activate='relu',
                 pe=None,
                 use_pytorch_dropout=True, dataset='weibo'):
        super().__init__()

        self.use_pytorch_dropout = use_pytorch_dropout
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        self.scaled = scaled
        self.ff_activate = ff_activate
        self.dropout = dropout
        self.ff_size = ff_size


        self.transformer_layer = TransformerEncoderLayer(hidden_size, num_heads,
                                                         layer_preprocess_sequence,
                                                         layer_postprocess_sequence,
                                                         dropout, scaled, ff_size,
                                                         max_seq_len=max_seq_len,
                                                         ff_activate=ff_activate,
                                                         use_pytorch_dropout=True,
                                                         dataset=dataset,
                                                         )

    def forward(self, query, key, value, seq_len):


        output = self.transformer_layer(query, key, value, seq_len)

        return output

