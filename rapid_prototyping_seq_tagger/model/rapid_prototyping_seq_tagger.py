
import sys
from logging import getLogger

from torch import nn as nn

from .word_rep_BiLSTM import BiLSTM
from .semicrf import SemiCRF

logger = getLogger(__name__)


class Rapid_prototyping_seq_tagger(nn.Module):
    def __init__(self, config_container, data_container):
        super(Rapid_prototyping_seq_tagger, self).__init__()
        # network definition
        if config_container.word_rep == 'BiLSTM':
            logger.debug('Initailize BiLSTM as word rep')
            self.word_rep = BiLSTM(config_container, data_container)
            self.word_rep_dim = config_container.lstm_hidden_dim
        else:
            logger.critical('Building word_rep error. word_rep can receive BiLSTM and BERT in config file')
            sys.exit(1)
        logger.debug('Set up semi-Markov CRF layer')
        self.semicrf = SemiCRF(
            self.word_rep_dim,
            data_container.get_label_vocab_size(),
            config_container.max_NE_length,
            data_container.pad_id)

    def forward(self, token_ids, char_ids, mask, multi_hot_label_tensor=None, calc_loss=False):
        """
        parameters
        ----------
            token_ids : tensor, size(batch size, maximum length of word tokens), dtype=torch.long
                input word level token indexes
            char_ids : tensor
                size(batch size, maximum length of word tokens, maximum length of char tokens), dtype=torch.long
                input character level token indexes
            mask : tensor, size(batch size, maximum length of word tokens), dtype=torch.uint8
                mask, 0 means padding
            multi_hot_label_tensor : tensor
                size(batch size, maximum length of word tokens, max NE length, label_size), dtype=torch.uint8
                multi hot label tensor
            calc_loss : bool
                contorol whether loss is calculated or not
        returns
        -------
            loss : tensor, size(1), dtype=torch.float
                loss, if calc_loss is False, then 0.0
            predict : tensor, size(batch size, maximum length of word tokens), dtype=torch.long
                predicted NE label
        """
        word_rep = self.word_rep(token_ids, char_ids, mask)
        loss, predict = self.semicrf(word_rep, mask, label_ids=multi_hot_label_tensor, calc_loss=calc_loss)

        return loss, predict
