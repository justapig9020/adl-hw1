from typing import Dict

import torch
from torch.nn import Embedding, LSTM, Linear, Softmax, GRU
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        rnn: str,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.rnn_input_dim = self.embed.embedding_dim
        self.rnn_output_dim = hidden_size
        if rnn == 'GRU':
            self.rnn = GRU(
                input_size = self.rnn_input_dim,
                hidden_size = self.rnn_output_dim,
                num_layers = num_layers,
                dropout = dropout,
                bidirectional = bidirectional,
                batch_first = True,
            )
        elif rnn == 'LSTM':
            self.rnn = LSTM(
                input_size = self.rnn_input_dim,
                hidden_size = self.rnn_output_dim,
                num_layers = num_layers,
                dropout = dropout,
                bidirectional = bidirectional,
                batch_first = True,
            )
        else:
            raise Exception('Unknow rnn')
        self.fc = Linear(self.rnn_output_dim, num_class)
        self.num_class = num_class
        self.sm = Softmax(dim=-1)

    @property
    def encoder_output_size(self) -> int:
        return self.embed.embedding_dim

    def forward(self, batch):
        text = batch['text']
        lens = batch['len']
        ev = self.embed(text)
        # packed = pack_padded_sequence(ev, lens, batch_first = True, enforce_sorted = False)
        output, _ = self.rnn(ev)
        # padded, _ = pad_packed_sequence(output, batch_first = True)
        output = self.fc(output[:, -1, :self.rnn_output_dim])
        output = self.sm(output)
        return output



class SeqTagger(SeqClassifier):
    def forward(self, batch):
        text = batch['tokens']
        len = batch['len']
        ev = self.embed(text)
        enc, _ = self.rnn(ev)
        # dec, _ = self.rnn(enc[:, :, :self.rnn_output_dim])
        output = self.fc(enc[:, :, :self.rnn_output_dim])
        output = self.sm(output)
        return output
