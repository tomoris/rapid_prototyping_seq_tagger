
from logging import getLogger

logger = getLogger(__name__)

import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class BiLSTM(nn.Module):
    def __init__(self, config_container, data_container):
        super(BiLSTM, self).__init__()
        if config_container.pretrained_word_emb_file is None:
            logger.debug('Set random vectors as initial word embedding')
            self.word_emb = nn.Embedding(
                data_container.get_word_vocab_size(),
                config_container.word_emb_dim,
                padding_idx=data_container.pad_id)
        else:
            logger.debug('Load pretrained word embedding')
            self.word_emb = nn.Embedding.from_pretrained(data_container.pretrained_word_emb, freeze=False)
        self.emb_dim = config_container.word_emb_dim
        self.dropout_word = nn.Dropout(config_container.dropout_rate)
        self.use_char_cnn = config_container.use_char_cnn
        if self.use_char_cnn:
            self.max_char_length = config_container.max_char_length
            self.char_emb_dim = config_container.char_emb_dim
            self.char_cnn_dim = config_container.char_cnn_dim
            self.char_cnn_window_size = config_container.char_cnn_window_size
            self.char_emb = nn.Embedding(
                data_container.get_char_vocab_size(),
                self.char_emb_dim,
                padding_idx=data_container.pad_id)
            self.char_cnn = nn.Conv1d(self.char_emb_dim, self.char_cnn_dim, self.char_cnn_window_size)
            self.char_pool = nn.MaxPool1d(self.max_char_length - (self.char_cnn_window_size - 1), stride=1)
            self.emb_dim += config_container.char_cnn_dim

        self.lstm = nn.LSTM(
            self.emb_dim,
            config_container.lstm_hidden_dim // 2,
            num_layers=config_container.lstm_layer,
            batch_first=True,
            bidirectional=True)

        self.dropout_lstm = nn.Dropout(config_container.dropout_rate)

    def forward(self, bacthed_word_data, bacthed_char_data, mask):
        """
        parameters
        ----------
            batched_word_data : tensor, size(batch size, maximum length of word tokens), dtype=torch.long
                input word level token indexes
            batched_char_data : tensor,
                size(batch size, maximum length of word tokens, maximum length of char tokens), dtype=torch.long
                input character level token indexes
            mask : tensor, size(batch size, maximum length of word tokens), dtype=torch.uint8
                mask, 0 means padding
        returns
        -------
            word_rep : tensor, size(batch size, maximum length of word tokens, hidden dim), dtype=torch.float
                each word representation
        """
        batch_size = bacthed_word_data.size(0)
        seq_len = bacthed_word_data.size(1)

        word_emb = self.word_emb(bacthed_word_data)
        if self.use_char_cnn:
            char_emb = self.char_emb(bacthed_char_data)
            char_cnn = self.char_cnn(
                char_emb.view(
                    batch_size * seq_len,
                    self.max_char_length,
                    self.char_emb_dim).transpose(
                    1,
                    2))
            char_cnn = self.char_pool(char_cnn)
            char_cnn = char_cnn.view(batch_size, seq_len, self.char_cnn_dim)
            word_emb = torch.cat([word_emb, char_cnn], dim=2)

        word_emb = self.dropout_word(word_emb)
        packed_word_emb = pack_padded_sequence(word_emb, torch.sum(mask, dim=1).long(), batch_first=True)
        word_rep, (_, _) = self.lstm(packed_word_emb)
        word_rep, _ = pad_packed_sequence(word_rep, batch_first=True)
        word_rep = self.dropout_lstm(word_rep)
        return word_rep
