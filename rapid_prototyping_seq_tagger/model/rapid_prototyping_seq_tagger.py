
from logging import getLogger
logger = getLogger(__name__)

import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn as nn

from .word_rep_BiLSTM import BiLSTM
from .semicrf import SemiCRF

try:
    import go  # pyhsmm with golang
    import bayselm  # pyhsmm with golang
except ImportError:
    logger.warning('import error! bayselm')




class Rapid_prototyping_seq_tagger(nn.Module):
    def __init__(self, config_container, data_container):
        super(Rapid_prototyping_seq_tagger, self).__init__()
        self.pad_str = '<pad>'
        self.pad_id = 0
        self.unk_str = '<unk>'
        self.unk_id = 1
        self.O_str = 'O'
        self.O_id = 1

        # network definition
        if config_container.word_rep == 'BiLSTM':
            logger.debug('Initialize BiLSTM as word rep')
            self.word_rep_net = BiLSTM(config_container, data_container)
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
        word_rep = self.word_rep_net(token_ids, char_ids, mask)
        if sents is not None and self.use_PYHSMM:
            each_score_in_PYHSMM, utf_char_sents = self._get_forward_score_of_PYHSMM(sents)
        else:
            each_score_in_PYHSMM, utf_char_sents = None, None
        loss, predict = self.semicrf(word_rep, mask, label_ids=multi_hot_label_tensor,
                                     calc_loss=calc_loss, each_score_in_PYHSMM=each_score_in_PYHSMM,
                                     utf_char_sents=utf_char_sents)

        return loss, predict

    def train_PYHSMM(self, data_container, threads, batch, joint_train=False):
        # ここはあとで pyhsmm がわの forwardScore を識別機のスコアを考慮したものに変更する必要がある。
        if joint_train:
            size = data_container.data_container_for_PYHSMM.Size
            rand_list = [j for j in range(size)]
            random.shuffle(rand_list)
            for i in range(0, size, batch):
                sent_string_list = []
                tokens_list = []
                token_ids_list = []
                lens_list = []
                for j in range(i, min(i + batch, size)):
                    r = rand_list[j]
                    sent_string = data_container.data_container_for_PYHSMM.GetSentString(r)
                    tokens = data_container.orig_data_for_PYHSMM[r]
                    t_ids = data_container.orig_data_word_ids_for_PYHSMM[r]
                    t_ids = torch.tensor(t_ids)
                    sent_string_list.append(sent_string)  # need to sort
                    tokens_list.append(tokens)
                    token_ids_list.append(t_ids)
                    lens_list.append(len(tokens))
                sorted_index = torch.argsort(torch.tensor(lens_list), descending=True).tolist()
                sorted_tokens_list = [tokens_list[b] for b in sorted_index]
                sorted_t_ids = [t_ids[b] for b in sorted_index]
                mask = [torch.ones(length, dtype=torch.long) for length in sorted_index]
                each_score_in_PYHSMM, utf_char_sents = self._get_forward_score_of_PYHSMM(sorted_tokens_list)
                token_ids = pad_sequence(sorted_t_ids, batch_first=True, padding_value=self.pad_id)
                mask = pad_sequence(mask, batch_first=True)
                device = torch.device('cuda')
                token_ids = token_ids.to(device)
                mask = mask.to(device)
                word_rep = self.word_rep_net(token_ids, None, mask)
                log_alpha, _ = self.calc_forward_alpha(
                    word_rep, mask, each_score_in_PYHSMM=each_score_in_PYHSMM, utf_char_sents=utf_char_sents)
                for b in range(len(sorted_index)):
                    sent = sent_string_list[sorted_index[b]]
                    log_alpha_list = log_alpha[b].view(-1)
                    self.pyhsmm.TrainWithDiscScore(sent, log_alpha_list, True)
            else:
                self.pyhsmm.TrainWordSegmentationAndPOSTagging(data_container.data_container_for_PYHSMM, threads, batch)
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
