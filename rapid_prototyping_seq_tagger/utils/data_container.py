
import random

import torch
from torch.nn.utils.rnn import pad_sequence

from . import convert_word_to_utf_char

import go  # pyhsmm with golang
import bayselm  # pyhsmm with golang

from logging import getLogger
logger = getLogger(__name__)


class Data_container(object):

    def __init__(self, config_container):
        self.pad_str = '<pad>'
        self.pad_id = 0
        self.unk_str = '<unk>'
        self.unk_id = 1
        self.O_str = 'O'
        self.O_id = 1

        self.w2i = {self.pad_str: self.pad_id, self.unk_str: self.unk_id}  # word index
        self.i2w = {self.pad_id: self.pad_str, self.unk_id: self.unk_str}
        self.c2i = {self.pad_str: self.pad_id, self.unk_str: self.unk_id}  # character index
        self.i2c = {self.pad_id: self.pad_str, self.unk_id: self.unk_str}
        self.l2i = {self.pad_str: self.pad_id, self.O_str: self.O_id}  # label index
        self.i2l = {self.pad_id: self.pad_str, self.O_id: self.O_str}
        self.semil2i = {self.pad_str: self.pad_id, self.O_str: self.O_id}  # semi-Markov label index
        self.i2semil = {self.pad_id: self.pad_str, self.O_id: self.O_str}

        self.semi_markov = config_container.semi_markov
        self.max_NE_length = config_container.max_NE_length

        self.max_char_length = 30
        if config_container.use_char_cnn:
            self.max_char_length = config_container.max_char_length
        self.norm_digit = config_container.norm_digit

        # token, token_ids, char_ids, label_ids, multi_hot_label_tensor
        self.data = {'train': [[], [], [], [], []], 'dev': [[], [], [], [], []], 'test': [[], [], [], [], []]}

        if config_container.pretrained_word_emb_file is not None:
            self.load_pretrained_word_emb(config_container.pretrained_word_emb_file, config_container.word_emb_dim)

        self.train_file_is_partially_labeled = False
        assert(config_container.train_file is not None)
        self.load_annotated_corpus(config_container.train_file, 'train', vocab_expansion=True)
        for label_str in self.l2i.keys():
            if label_str != self.O_str and label_str != self.pad_str:
                label_stem = label_str.split('-')[1]
                if label_stem not in self.semil2i:
                    self.semil2i[label_stem] = len(self.semil2i)
                    self.i2semil[len(self.i2semil)] = label_stem
        for label_stem in self.semil2i.keys():
            if label_stem != self.O_str and label_stem != self.pad_str:
                for BIES in ['B', 'I', 'E', 'S']:
                    label_str_tmp = BIES + '-' + label_stem
                    if label_str_tmp not in self.l2i:
                        self.l2i[label_str_tmp] = len(self.l2i)
                        self.i2l[len(self.i2l)] = label_str_tmp
        for i, label_ids in enumerate(self.data['train'][3]):
            assert(i == len(self.data['train'][4]))
            self.data['train'][4].append(self._adjust_label_id_to_semi_markov_multi_label_tensor(label_ids))

        if config_container.pretrained_word_emb_file is not None:
            if self.pretrained_word_emb.size(0) != len(self.w2i):
                logger.debug('number of out of pretrained vocab {}'.format(
                    len(self.w2i) - self.pretrained_word_emb.size(0)))
                pretrained_embedding_list = [self.pretrained_word_emb]
                for i in range(self.pretrained_word_emb.size(0), len(self.w2i)):
                    rand_vec = torch.rand(config_container.word_emb_dim) * 2 \
                        * torch.sqrt(torch.tensor([3.0 / config_container.word_emb_dim]))
                    rand_vec = rand_vec \
                        - torch.sqrt(torch.tensor(
                            [3.0 / config_container.word_emb_dim])).expand(config_container.word_emb_dim)
                    pretrained_embedding_list.append(rand_vec.unsqueeze(0))
                self.pretrained_word_emb = torch.cat(pretrained_embedding_list, dim=0)
            assert(self.pretrained_word_emb.size(0) == len(self.w2i))

        if config_container.dev_file is not None:
            self.load_annotated_corpus(config_container.dev_file, 'dev', vocab_expansion=False)
            for i, label_ids in enumerate(self.data['dev'][3]):
                assert(i == len(self.data['dev'][4]))
                self.data['dev'][4].append(self._adjust_label_id_to_semi_markov_multi_label_tensor(label_ids))
        if config_container.test_file is not None:
            self.load_annotated_corpus(config_container.test_file, 'test', vocab_expansion=False)
            for i, label_ids in enumerate(self.data['test'][3]):
                assert(i == len(self.data['test'][4]))
                self.data['test'][4].append(self._adjust_label_id_to_semi_markov_multi_label_tensor(label_ids))

        logger.debug('label vocab {}'.format(self.l2i))
        logger.debug('semilabel vocab {}'.format(self.semil2i))

        if config_container.use_PYHSMM:
            self._load_data_container(config_container)

    def load_annotated_corpus(self, file_name, data_type, vocab_expansion=True):
        """
        load annotated corpus. supports partially annotated data
        annotated data form is that each token is separated by \n and each line is
                (token + tab + label_1 + label_2 + ...)
            if label columnn is null then all possible label is used
        ----------
        parameters
        ----------
            file_name : str
            data_type : str, 'train' or 'dev' or test'
            vocab_expansion : bool
                if true then when a key not in dicts (self.w2i, self.c2i, self.l2i, self.semil2i), the dict is expanded
        returns
        -------
            None
        """
        tokens = []
        token_ids = []
        char_ids = []
        label_ids = []
        for line in open(file_name):
            line = line.rstrip()
            if line == '':
                assert(len(tokens) != 0)
                assert(len(tokens) == len(token_ids))
                assert(len(tokens) == len(char_ids))
                assert(len(tokens) == len(label_ids))
                self.data[data_type][0].append(tokens)
                self.data[data_type][1].append(torch.tensor(token_ids))
                self.data[data_type][2].append(torch.tensor(char_ids))
                self.data[data_type][3].append(label_ids)
                tokens = []
                token_ids = []
                char_ids = []
                label_ids = []
                continue
            line_sp = line.split('\t')
            token = line_sp[0]
            if self.norm_digit:
                token = ''.join(['0' if _.isdigit() else _ for _ in token])
            tokens.append(token)
            if token not in self.w2i:
                if vocab_expansion:
                    if token.lower() not in self.w2i:
                        self.w2i[token] = len(self.w2i)
                        self.i2w[len(self.i2w)] = token
                        token_id = self.w2i[token]
                    else:
                        token_id = self.w2i[token.lower()]
                else:
                    if token.lower() in self.w2i:
                        token_id = self.w2i[token.lower()]
                    else:
                        token_id = self.unk_id
            else:
                token_id = self.w2i[token]
            token_ids.append(token_id)
            char_ids_item = []
            for c in token:
                if c not in self.c2i:
                    if vocab_expansion:
                        self.c2i[c] = len(self.c2i)
                        self.i2c[len(self.i2c)] = c
                        char_id = self.c2i[c]
                    else:
                        char_id = self.unk_id
                else:
                    char_id = self.c2i[c]
                char_ids_item.append(char_id)
            # assert(len(char_ids_item) <= self.max_char_length)
            if len(char_ids_item) <= self.max_char_length:
                char_ids_item = char_ids_item + [self.pad_id for _ in range(self.max_char_length - len(char_ids_item))]
            else:
                char_ids_item = char_ids_item[:self.max_char_length]
            char_ids.append(char_ids_item)
            label_ids.append([])
            for label in line_sp[1:]:
                if label not in self.l2i:
                    if vocab_expansion is False:
                        assert(False)
                    self.l2i[label] = len(self.l2i)
                    self.i2l[len(self.i2l)] = label
                    label_id = self.l2i[label]
                else:
                    label_id = self.l2i[label]
                label_ids[-1].append(label_id)
            if len(label_ids[-1]) == 0:
                label_ids[-1].append(-1)
            if label_ids[-1] == [-1] or len(label_ids[-1]) >= 2:
                assert(data_type == 'train')
                self.train_file_is_partially_labeled = True

        if len(tokens) != 0:
            assert(len(tokens) == len(token_ids))
            assert(len(tokens) == len(char_ids))
            assert(len(tokens) == len(label_ids))
            self.data[data_type][0].append(tokens)
            self.data[data_type][1].append(torch.tensor(token_ids))
            self.data[data_type][2].append(torch.tensor(char_ids))
            self.data[data_type][3].append(label_ids)

        assert(len(self.data[data_type][0]) == len(self.data[data_type][1]))
        assert(len(self.data[data_type][0]) == len(self.data[data_type][2]))
        assert(len(self.data[data_type][0]) == len(self.data[data_type][3]))

    def adjust_batched_data(self, batched_data):
        # token_ids, char_ids, label_ids, multi_hot_label_tensor, masks
        adjusted_batched_data = [[] for _ in range(len(batched_data))]
        adjusted_batched_data[0] = pad_sequence(batched_data[0], batch_first=True, padding_value=self.pad_id)
        adjusted_batched_data[1] = pad_sequence(batched_data[1], batch_first=True, padding_value=self.pad_id)
        adjusted_batched_data[2] = batched_data[2]
        # adjusted_batched_data[2] = pad_sequence(batched_data[2], batch_first=True)
        adjusted_batched_data[3] = pad_sequence(batched_data[3], batch_first=True)
        adjusted_batched_data[4] = pad_sequence(batched_data[4], batch_first=True)
        adjusted_batched_data[5] = batched_data[5]
        return adjusted_batched_data

    def __iter__(self):
        return self

    def __next__(self):
        if self._start_idx_for_iter >= self._data_len_for_iter:
            raise StopIteration()
        # token_ids, char_ids, label_ids, multi_hot_tensor, masks, tokens
        batched_data = [[], [], [], [], [], []]
        batched_data_lens = []
        for i in range(self._start_idx_for_iter,
                       min(self._start_idx_for_iter + self._batch_size_for_iter, self._data_len_for_iter)):
            r = self._rand_list_for_iter[i]
            batched_data_lens.append(len(self._data_for_iter[0][r]))
            for j in range(1, len(self._data_for_iter)):
                batched_data[j - 1].append(self._data_for_iter[j][r])
            # mask
            batched_data[4].append(torch.tensor([1 for _ in range(len(self._data_for_iter[0][r]))], dtype=torch.uint8))
            # tokens
            batched_data[5].append(self._data_for_iter[0][r])
        batched_data_lens = torch.argsort(torch.tensor(batched_data_lens), descending=True)

        sorted_batched_data = [[None for __ in range(len(batched_data[0]))]
                               for _ in range(len(self._data_for_iter) + 1)]
        for j in range(len(batched_data)):
            for i in range(len(batched_data[j])):
                sorted_batched_data[j - 1][i] = batched_data[j - 1][batched_data_lens[i]]

        adjusted_batched_data = self.adjust_batched_data(sorted_batched_data)

        self._start_idx_for_iter += self._batch_size_for_iter
        return adjusted_batched_data

    def __call__(self, data_type, batch_size):
        self._data_type_for_iter = data_type
        self._batch_size_for_iter = batch_size
        self._start_idx_for_iter = 0
        self._data_for_iter = self.data[self._data_type_for_iter]
        self._data_len_for_iter = len(self.data[self._data_type_for_iter][0])
        self._rand_list_for_iter = [i for i in range(self._data_len_for_iter)]
        if data_type == 'train':
            random.shuffle(self._rand_list_for_iter)
        return self

    def _adjust_label_id_to_semi_markov_multi_label_tensor(self, label_ids):
        """
        parameters
        ----------
            label_ids : list, len(label_ids) is length of tokens, len(label_ids[t]) is number of possible label
                target NE label indexes
                -1 means all possible labels
        returns
        -------
            semi_markov_multi_label_tensor : tensor,
                size(length of tokens, maximum NE length, label size), dtype=torch.long
                target NE label indexes as multi hot vector
                supports partailly labeld data
        """
        if self.semi_markov:
            semi_markov_multi_label_tensor = torch.zeros(len(label_ids), self.max_NE_length,
                                                         len(self.semil2i), dtype=torch.long)
            for t in range(len(label_ids)):
                for k in range(self.max_NE_length):
                    if k == 0:
                        if label_ids[t] == -1:
                            possible_semi_markov_labels = {_ for _ in self.semil2i.keys() if _ != self.pad_str}
                            possible_semi_markov_labels_prev = {_ for _ in self.semil2i.keys() if _ != self.pad_str}
                        else:
                            possible_semi_markov_labels = set()
                            possible_semi_markov_labels_prev = set()
                            label_set_at_t_minus_k = [_ for _ in label_ids[t - k]]
                            for label_id in label_set_at_t_minus_k:
                                if label_id == -1 or label_id == self.pad_id:
                                    assert(False)
                                label_str = self.i2l[label_id]
                                if label_str == self.O_str:
                                    possible_semi_markov_labels.add(label_str)
                                    possible_semi_markov_labels_prev.add(label_str)
                                else:
                                    label_stem = label_str.split('-')[1]
                                    BIES = label_str.split('-')[0]
                                    if BIES == 'E':
                                        possible_semi_markov_labels_prev.add(label_stem)
                                    elif BIES == 'S':
                                        possible_semi_markov_labels.add(label_stem)
                        # multi hot vector
                        for l in possible_semi_markov_labels:
                            semi_markov_multi_label_tensor[t, k, self.semil2i[l]] = 1
                    elif t - k >= 0:
                        possible_semi_markov_labels_at_t_minus_k = set()
                        if label_ids[t - k] == -1:
                            possible_semi_markov_labels_at_t_minus_k = \
                                {_ for _ in self.semil2i.keys() if _ != self.pad_str and _ != self.O_str}
                        else:
                            label_set_at_t_minus_k = [_ for _ in label_ids[t - k]]
                            for label_id in label_set_at_t_minus_k:
                                if label_id == -1 or label_id == self.pad_id:
                                    assert(False)
                                label_str = self.i2l[label_id]
                                if label_str == self.O_str:
                                    pass
                                else:
                                    label_stem = label_str.split('-')[1]
                                    BIES = label_str.split('-')[0]
                                    if BIES == 'B':
                                        possible_semi_markov_labels_at_t_minus_k.add(label_stem)
                        # multi hot vector
                        possible_semi_markov_labels = \
                            possible_semi_markov_labels_at_t_minus_k & possible_semi_markov_labels_prev
                        for l in possible_semi_markov_labels:
                            semi_markov_multi_label_tensor[t, k, self.semil2i[l]] = 1
                        # remove inconsistent label path
                        for label_stem in self.semil2i.keys():
                            if label_stem + 'I' not in label_set_at_t_minus_k:
                                possible_semi_markov_labels_prev - {label_stem + 'I'}
                    else:
                        possible_semi_markov_labels = {self.pad_str}
                        # multi hot vector
                        for l in possible_semi_markov_labels:
                            semi_markov_multi_label_tensor[t, k, self.semil2i[l]] = 1
        else:
            semi_markov_multi_label_tensor = torch.zeros(len(label_ids), self.max_NE_length,
                                                         len(self.l2i), dtype=torch.long)
            for t in range(len(label_ids)):
                if label_ids[t] == [-1]:
                    semi_markov_multi_label_tensor[t, 0, :] = 1
                    semi_markov_multi_label_tensor[t, 0, self.pad_id] = 0
                else:
                    for label_id in label_ids[t]:
                        assert(label_id != self.pad_id)
                        assert(label_id != -1)
                        semi_markov_multi_label_tensor[t, 0, label_id] = 1
        return semi_markov_multi_label_tensor

    def load_pretrained_word_emb(self, pretrained_word_emb_file, word_emb_dim):
        pretrained_embedding_list = []
        pad_vec = torch.zeros(1, word_emb_dim)
        unk_vec = torch.zeros(1, word_emb_dim)
        pretrained_embedding_list.append(pad_vec)
        pretrained_embedding_list.append(unk_vec)
        count = 0
        for line in open(pretrained_word_emb_file, 'r'):
            line = line.rstrip()
            line_sp = line.split(' ')
            token = line_sp[0]
            vec = [float(_) for _ in line_sp[1:]]
            assert(len(vec) == word_emb_dim)
            self.w2i[token] = len(self.w2i)
            self.i2w[len(self.i2w)] = token
            vec = torch.tensor(vec)
            vec = vec / torch.sqrt(torch.sum(torch.pow(vec, 2)))
            vec = vec.unsqueeze(0)
            pretrained_embedding_list.append(vec)
            unk_vec = unk_vec + vec
            count += 1
        unk_vec = unk_vec / float(count)
        pretrained_embedding_list[1] = unk_vec
        assert(len(pretrained_embedding_list) == len(self.w2i))
        self.pretrained_word_emb = torch.cat(pretrained_embedding_list, dim=0)

    def get_word_vocab_size(self):
        return len(self.w2i)

    def get_char_vocab_size(self):
        return len(self.c2i)

    def get_label_vocab_size(self):
        if self.semi_markov:
            return len(self.semil2i)
        else:
            return len(self.l2i)

    def get_id2label(self):
        if self.semi_markov:
            @property
            def i2semil(self):
                return self.i2semil
            return self.i2semil
        else:
            @property
            def i2l(self):
                return self.i2l
            return self.i2l

    def i2l(self):
        return self.i2l

    def _load_data_container(self, config_container):
        sent_list = []
        sent_word_ids_list = []
        sent_utf_char_list = []
        self.w2c = convert_word_to_utf_char.Word_to_Char()
        for line in open(config_container.PYHSMM_train_file):
            line = line.rstrip()
            line_sp = line.split(' ')
            newsent = ''
            orig_words = []
            orig_word_ids = []
            for token in line_sp:
                self.w2c.add(token)
                newsent += self.w2c.get(token)
                orig_words.append(token)
                orig_word_ids.append(self.w2i.get(token, self.unk_id))
            sent_utf_char_list.append(newsent)
            sent_list.append(orig_words)
            sent_word_ids_list.append(orig_word_ids)

        slice_sent = go.Slice_string(sent_utf_char_list)
        self.data_container_for_PYHSMM = bayselm.NewDataContainerFromSents(slice_sent)
        self.orig_data_for_PYHSMM = sent_list
        self.orig_data_word_ids_for_PYHSMM = sent_word_ids_list
        self.sent_utf_char_list = sent_utf_char_list

        return
