
from logging import getLogger

import torch
from torch import nn as nn

logger = getLogger(__name__)

class SemiCRF(nn.Module):
    def __init__(self, word_rep_dim, label_size, max_NE_length, pad_id):
        super(SemiCRF, self).__init__()
        self.label_size = label_size
        self.max_NE_length = max_NE_length
        self.word_rep_dim = word_rep_dim
        if max_NE_length >= 2:
            self.cnn = nn.ModuleList([nn.Conv1d(self.word_rep_dim, self.word_rep_dim, k+1) for k in range(self.max_NE_length)])
        self.classifier = nn.Linear(self.word_rep_dim, self.label_size)

        # Note: the parameters are if you use them as defferent vectors repeatedly, use Tensor.clone()
        self.zero_score = nn.Parameter(torch.zeros(1, 1))
        self.zero_score.requires_grad = False
        self.log_zero_score = nn.Parameter(torch.ones(1, 1) * -10000.0)
        self.log_zero_score.requires_grad = False
        self.log_alpha_item = nn.Parameter(torch.ones(1, 1, self.label_size) * -10000.0)
        self.log_alpha_item.requires_grad = False
        self.one_vec = nn.Parameter(torch.ones(1, 1))
        self.one_vec.requires_grad = False

        self.pad_id = pad_id
        init_T = torch.zeros(1, self.label_size, self.label_size)
        init_T[0, self.pad_id, :] = -10000.0
        init_T[0, :, self.pad_id] = -10000.0
        self.T = nn.Parameter(init_T)  # T[j, i] represents transition score i to j
        init_T_from_BEGIN = torch.zeros(1, self.label_size)
        init_T_to_END = torch.zeros(1, self.label_size)
        init_T_from_BEGIN[:, self.pad_id] = -10000.0
        init_T_to_END[:, self.pad_id] = -10000.0
        self.T_from_BEGIN = nn.Parameter(init_T_from_BEGIN)
        self.T_to_END = nn.Parameter(init_T_to_END)

    def forward(self, word_rep, mask, label_ids=None, calc_loss=False):
        """
        parameters
        ----------
            word_rep : tensor, size(batch size, maximum length of word tokens, hidden dim), dtype=torch.float
                each word representation
            mask : tensor, size(batch size, maximum length of word tokens), dtype=torch.uint8
                mask, 0 means padding
            label_ids : tensor, size(batch size, maximum length of word tokens), dtype=torch.long
                target NE label indexes
            calc_loss : bool
                contorol whether loss is calculated or not
        returns
        -------
            loss : tensor, size(1), dtype=torch.float
                semi-Markov CRF loss = -log( score(y_gold|x) / sum(y' in Y, score(y'|x) ), if calc_loss is False, then 0.0
                if calc_loss is False, then 0.0
            predict_triplets_list : list, len(predict_triplets_list) is batch size, len(predict_triplets_list[b]) is number of triplets
                predicted NE label
                predict_triplets_list[b][t] represents triplet (start word position, end word position, label id) of t-th triplet in b-th batch data
                predict_triplets_list[b][0][0] is 0, predict_triplets_list[b][-1][0] is length of the sentence
        """
        batch_size = word_rep.size(0)
        seq_len = word_rep.size(1)

        label_score = self.classifier(word_rep)
        label_score[:, :, self.pad_id] = -10000.0
        if self.max_NE_length == 1:
            label_score = self.classifier(word_rep)
            label_score[:, :, self.pad_id] = -10000.0
            label_score = label_score.unsqueeze(2)
        else:
            word_seq_rep = self.zero_score.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, self.max_NE_length, self.word_rep_dim).clone()
            for k in range(self.max_NE_length):
                if seq_len >= (k+1):
                    word_seq_rep_tmp = self.cnn[k](word_rep.transpose(1, 2))
                    word_seq_rep[:, k:, k, :] = word_seq_rep_tmp.transpose(1, 2)
            label_score = self.classifier(word_seq_rep)
            label_score[:, :, :, self.pad_id] = -10000.0

        mask = mask.unsqueeze(2).expand(batch_size, seq_len, self.max_NE_length).clone()
        for k in range(self.max_NE_length):
            if seq_len >= (k+1):
                mask[:, k, k+1:] = 0
        label_score = torch.where((mask == 1).unsqueeze(3).expand(batch_size, seq_len, self.max_NE_length, self.label_size), label_score, self.log_zero_score.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, self.max_NE_length, self.label_size).clone())

        log_alpha = self._semicrf_forward(label_score, mask)

        if calc_loss:
            batch_size = label_score.size(0)
            seq_len = label_score.size(1)

            log_gold_path_score = self._calc_log_gold_path_score(label_score, mask, label_ids)
            # calculate log gold path score in last token position
            log_alpha_tmp_list = []
            last_token_idx = (torch.sum(mask[:, :, 0].long(), dim=1)).unsqueeze(1).unsqueeze(1).expand(batch_size, seq_len, self.label_size)
            padding_log_alpha_idx = self.one_vec.unsqueeze(0).expand(batch_size, seq_len, self.label_size).clone().long() * (seq_len -1)
            for k in range(self.max_NE_length):
                last_token_idx_k = last_token_idx - (k+1)
                # if last_token_id >= 0 then use padding value in log_gold_alpha
                last_token_idx_k = torch.where(last_token_idx_k >= 0, last_token_idx_k, padding_log_alpha_idx)
                log_alpha_tmp = torch.gather(log_alpha[:, :, k, :], 1, last_token_idx_k)[:, 0, :]
                log_alpha_tmp = log_alpha_tmp + self.T_to_END.expand(batch_size, -1)
                log_alpha_tmp = log_alpha_tmp.unsqueeze(1)
                log_alpha_tmp_list.append(log_alpha_tmp)
            log_alpha_tmp = torch.cat(log_alpha_tmp_list, dim=2)
            log_all_path_score = self._log_sum_exp(log_alpha_tmp)
            log_all_path_score = log_all_path_score.squeeze()

            loss = log_all_path_score - log_gold_path_score
            loss = loss.mean()
            assert(loss.item() >= 0.0)
        else:
            loss = 0.0

        predict = self._predict(label_score, mask, log_alpha)
        return loss, predict

    def _semicrf_forward(self, label_score, mask):
        """
        parameters
        ----------
            label_score : tensor, size(batch size, maximum length of word tokens, hidden dim), dtype=torch.float
                each label score (log)
            mask : tensor, size(batch size, maximum length of word tokens), dtype=torch.uint8
                mask, 0 means padding
        returns
        -------
            log_alpha : tensor, size(batch size, maximum length of word tokens +1, self.max_NE_length, label size), dtype=torch.float
                semi-Markov crf forward score, log_alpha[:, t, k, l] represents l-th label score of token sequence from (t-k)-th to t-th token
                last value in dim=1 is padding need to calc. all path score
        """
        batch_size = label_score.size(0)
        seq_len = label_score.size(1)
        log_alpha = self.log_alpha_item.expand(batch_size, seq_len +1, self.max_NE_length, self.label_size).clone()
        log_zero_score = self.log_zero_score.expand(batch_size, self.label_size).clone()

        # log_alpha[b][t][k][l]
        for t in range(seq_len):
            for k in range(self.max_NE_length):
                if (t - k) == 0: # if previous label is BEGIN
                    log_alpha[:, t, k, :] = label_score[:, t, k, :] + self.T_from_BEGIN.expand(batch_size, -1)
                elif (t - k) < 0:
                    pass
                else:
                    # make log_alpha_tmp size(batch size, label size, max NE length * label size), dim=2 is summed up in logsumexp later
                    log_alpha_tmp_list = []
                    for kk in range(self.max_NE_length):
                        log_alpha_tmp = label_score[:, t, k, :].unsqueeze(2).expand(-1, -1, self.label_size) + self.T.expand(batch_size, -1, -1) + log_alpha[:, t-(k+1), kk, :].unsqueeze(1).expand(-1, self.label_size, -1)
                        log_alpha_tmp_list.append(log_alpha_tmp)
                    log_alpha_tmp = torch.cat(log_alpha_tmp_list, dim=2)
                    logsumexp_alpha = self._log_sum_exp(log_alpha_tmp)
                    # if padding token then forward score log(0.0)
                    log_alpha[:, t, k, :] = torch.where((mask[:, t, k] == 1).unsqueeze(1).expand(-1, self.label_size), logsumexp_alpha, log_zero_score)

        return log_alpha

    def _calc_log_gold_path_score(self, label_score, mask, label_ids):
        """
        parameters
        ----------
            label_score : tensor, size(batch size, maximum length of word tokens, hidden dim), dtype=torch.float
                each label score (log)
            mask : tensor, size(batch size, maximum length of word tokens), dtype=torch.uint8
                mask, 0 means padding
            label_ids : tensor, size(batch size, maximum length of word tokens), dtype=torch.long
                target NE label indexes
        returns
        -------
            log_gold_path_score : size(batch size), dtype=torch.float
                semi-Markov gold path score, gold_path_score[b] represents b-th gold path score
                it can be partially gold path score if label_ids is multi label
        """
        batch_size = label_score.size(0)
        seq_len = label_score.size(1)
        log_gold_alpha = self.log_alpha_item.expand(batch_size, seq_len +1, self.max_NE_length, self.label_size).clone()
        log_zero_score = self.log_zero_score.expand(batch_size, self.label_size).clone()
        zero_score = self.zero_score.expand(batch_size, self.label_size).clone()

        # log_gold_alpha[b][t][k][l]
        for t in range(seq_len):
            for k in range(self.max_NE_length):
                if (t - k) == 0: # if previous label is BEGIN
                    log_gold_alpha[:, t, k, :] = label_score[:, t, k, :] + self.T_from_BEGIN.expand(batch_size, -1)
                    # replace log(0.0) if l-th label is not gold label in token sequence from (t-k)-th token to t-th token in b-th batch data
                    multi_hot_label_vector = label_ids[:, t, k, :]
                    # assert(multi_hot_label_vector.sum().item() != 0.0)
                    log_gold_alpha[:, t, k, :] = torch.where(multi_hot_label_vector == 1, log_gold_alpha[:, t, k, :], log_zero_score)
                elif (t - k) < 0:
                    pass
                else:
                    # make log_alpha_tmp size(batch size, label size, max NE length * label size),  dim=2 is summed up in logsumexp later
                    log_alpha_tmp_list = []
                    for kk in range(self.max_NE_length):
                        log_alpha_tmp = label_score[:, t, k, :].unsqueeze(2).expand(-1, -1, self.label_size) + self.T.expand(batch_size, -1, -1) + log_gold_alpha[:, t-(k+1), kk, :].unsqueeze(1).expand(-1, self.label_size, -1)
                        log_alpha_tmp_list.append(log_alpha_tmp)
                    log_alpha_tmp = torch.cat(log_alpha_tmp_list, dim=2)
                    logsumexp_alpha = self._log_sum_exp(log_alpha_tmp)
                    multi_hot_label_vector = label_ids[:, t, k, :]
                    # assert(multi_hot_label_vector.sum().item() != 0.0)
                    log_gold_alpha[:, t, k, :] = torch.where(multi_hot_label_vector == 1, logsumexp_alpha, log_zero_score)
                    # if padding token then forward score log(0.0)
                    log_gold_alpha[:, t, k, :] = torch.where((mask[:, t, k] == 1).unsqueeze(1).expand(-1, self.label_size), log_gold_alpha[:, t, k, :], log_zero_score)
        # calculate log gold path score in last token position
        log_alpha_tmp_list = []
        last_token_idx = (torch.sum(mask[:, :, 0].long(), dim=1)).unsqueeze(1).unsqueeze(1).expand(batch_size, seq_len, self.label_size)
        padding_log_alpha_idx = self.one_vec.unsqueeze(0).expand(batch_size, seq_len, self.label_size).clone().long() * (seq_len -1)
        for k in range(self.max_NE_length):
            last_token_idx_k = last_token_idx - (k+1)
            # if last_token_id >= 0 then use padding value in log_gold_alpha
            last_token_idx_k = torch.where(last_token_idx_k >= 0, last_token_idx_k, padding_log_alpha_idx)
            log_alpha_tmp = torch.gather(log_gold_alpha[:, :, k, :], 1, last_token_idx_k)[:, 0, :]
            log_alpha_tmp = log_alpha_tmp + self.T_to_END.expand(batch_size, -1)
            log_alpha_tmp = log_alpha_tmp.unsqueeze(1)
            log_alpha_tmp_list.append(log_alpha_tmp)
        log_alpha_tmp = torch.cat(log_alpha_tmp_list, dim=2)
        log_gold_path_score = self._log_sum_exp(log_alpha_tmp)
        log_gold_path_score = log_gold_path_score.squeeze()

        return log_gold_path_score

    def _predict(self, label_score, mask, log_alpha):
        """
        parameters
        ----------
            label_score : tensor, size(batch size, maximum length of word tokens, maximun NE length, label size), dtype=torch.float
                each label score (log)
            mask : tensor, size(batch size, maximum length of word tokens), dtype=torch.uint8
                mask, 0 means padding
            log_alpha : tensor, size(batch size, maximum length of word tokens +1, self.max_NE_length, label size), dtype=torch.float
                semi-Markov crf forward score, log_alpha[:, t, k, l] represents l-th label score of token sequence from (t-k)-th to t-th token
                last value in dim=1 is padding need to calc. all path score
        returns
        -------
            predict_triplets_list : list, len(predict_triplets_list) is batch size, len(predict_triplets_list[b]) is number of triplets
                predicted NE label
                predict_triplets_list[b][t] represents triplet (start word position, end word position, label id) of t-th triplet in b-th batch data
                predict_triplets_list[b][0][0] is 0, predict_triplets_list[b][-1][0] is length of the sentence
        """
        batch_size = label_score.size(0)
        seq_len = label_score.size(1)

        predict_list = []
        current_token_idx = (torch.sum(mask[:, :, 0].long(), dim=1)).unsqueeze(1).unsqueeze(1).expand(batch_size, seq_len, self.label_size)
        current_label_T = self.T_to_END.expand(batch_size, -1)
        pad_ids = self.one_vec.clone().expand(batch_size, self.label_size) * self.pad_id
        begin_position_ids = self.one_vec.clone().expand(batch_size, seq_len, self.label_size).long() * 0
        padding_position_ids = self.one_vec.clone().expand(batch_size, seq_len, self.label_size).long() * seq_len

        while ((current_token_idx[:, 0, 0] == 0).sum().item() != batch_size):
            max_score_list = []
            max_score_label_id_list = []
            for k in range(self.max_NE_length):
                log_alpha_tmp = torch.gather(log_alpha[:, :, k, :], 1, torch.where(current_token_idx - (k+1) >= 0, current_token_idx - (k+1), padding_position_ids))[:, 0, :]
                log_alpha_tmp = log_alpha_tmp + current_label_T
                max_score_in_k, max_score_label_id_in_k = torch.max(log_alpha_tmp, dim=1)
                max_score_list.append(max_score_in_k.unsqueeze(0))
                max_score_label_id_list.append(max_score_label_id_in_k.unsqueeze(1))
            max_score = torch.cat(max_score_list, dim=0)
            max_score_label_id = torch.cat(max_score_label_id_list, dim=1)
            _, max_k = torch.max(max_score, dim=0)
            max_label_id = torch.gather(max_score_label_id, 1, max_k.unsqueeze(1))[:, 0]

            prev_token_idx = current_token_idx - (max_k.unsqueeze(1).unsqueeze(1).expand(batch_size, seq_len, self.label_size) + 1)
            predict_list.append(torch.cat([prev_token_idx[:, 0, 0].unsqueeze(1), current_token_idx[:, 0, 0].unsqueeze(1) -1, max_label_id.unsqueeze(1)], dim=1))

            current_token_idx = prev_token_idx
            current_label_T = self.T[0, max_label_id]
            # padding
            current_label_T = torch.where(current_token_idx[:, 0, :] > 0, current_label_T, pad_ids)
            current_token_idx = torch.where(current_token_idx > 0, current_token_idx, begin_position_ids)

        # adjust predict format
        predict_triplets_list = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            for predict in predict_list:
                if predict[b][0] >= 0:
                    predict_triplets_list[b].append(predict[b].tolist())
            predict_triplets_list[b].reverse()
        return predict_triplets_list

    def _log_sum_exp(self, alpha):
        """
        parameters
        ----------
            alpha : tensor, size(batch size, label size, any(vanishing)), dtype=torch.float
        returns
        -------
            log_alpha_item : tensor, size(batch size, label size), dtype=torch.float
        """
        max_score, _ = torch.max(alpha, dim=2)
        log_alpha_item = max_score + torch.log(torch.sum(torch.exp(alpha - max_score.unsqueeze(2).expand_as(alpha)), dim=2))
        return log_alpha_item
