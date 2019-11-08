
from logging import getLogger

from torch import nn as nn

from .word_rep_BiLSTM import BiLSTM
from .semicrf import SemiCRF

import go  # pyhsmm with golang
import bayselm  # pyhsmm with golang

logger = getLogger(__name__)


class Rapid_prototyping_seq_tagger(nn.Module):
    def __init__(self, config_container, data_container):
        super(Rapid_prototyping_seq_tagger, self).__init__()
        # network definition
        if config_container.word_rep == 'BiLSTM':
            logger.debug('Initialize BiLSTM as word rep')
            self.word_rep = BiLSTM(config_container, data_container)
            self.word_rep_dim = config_container.lstm_hidden_dim
        else:
            logger.critical('Building word_rep error. word_rep can receive BiLSTM and BERT in config file')
            raise NotImplementedError
        logger.debug('Set up semi-Markov CRF layer')
        self.semicrf = SemiCRF(self.word_rep_dim, data_container.get_label_vocab_size(),
                               config_container.max_NE_length, data_container.pad_id)

        self.use_PYHSMM = config_container.use_PYHSMM
        if self.use_PYHSMM:
            self.pyhsmm = bayselm.NewPYHSMM(config_container.PYHSMM_theta,
                                            config_container.PYHSMM_d,
                                            config_container.PYHSMM_gammaA,
                                            config_container.PYHSMM_gammaB,
                                            config_container.PYHSMM_betaA,
                                            config_container.PYHSMM_betaB,
                                            config_container.PYHSMM_alpha,
                                            config_container.PYHSMM_beta,
                                            config_container.PYHSMM_maxNgram,
                                            config_container.PYHSMM_maxWordLength,
                                            data_container.get_label_vocab_size())
            self.pyhsmm.Initialize(data_container.data_container_for_PYHSMM)
            self.w2c = data_container.w2c
            self.threads = config_container.PYHSMM_threads

    def forward(self, token_ids, char_ids, mask, multi_hot_label_tensor=None,
                calc_loss=False, sents=None):
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
        if sents is not None and self.use_PYHSMM:
            each_score_in_PYHSMM, utf_char_sents = self._get_forward_score_of_PYHSMM(sents)
        else:
            each_score_in_PYHSMM, utf_char_sents = None, None
        loss, predict = self.semicrf(word_rep, mask, label_ids=multi_hot_label_tensor,
                                     calc_loss=calc_loss, each_score_in_PYHSMM=each_score_in_PYHSMM,
                                     utf_char_sents=utf_char_sents)

        return loss, predict

    def train_PYHSMM(self, data_container_for_PYHSMM, threads, batch):
        # ここはあとで pyhsmm がわの forwardScore を識別機のスコアを考慮したものに変更する必要がある。
        self.pyhsmm.TrainWordSegmentationAndPOSTagging(data_container_for_PYHSMM, threads, batch)
        #
        return

    def _get_forward_score_of_PYHSMM(self, sents):
        utf_char_sents = []
        for sent in sents:
            new_utf_char_sent = ''
            for token in sent:
                new_utf_char_sent += self.w2c.get(token)
            utf_char_sents.append(new_utf_char_sent)

        slice_sent = go.Slice_string(utf_char_sents)
        each_score = self.pyhsmm.GetEachScore(slice_sent, self.threads)

        return each_score, utf_char_sents
