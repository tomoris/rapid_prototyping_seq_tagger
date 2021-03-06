

# model configuration and hyper-parameters
class Config():
    def __init__(self):
        # load files
        self.pretrained_word_emb_file = None
        self.train_file = './sample/train.txt'
        self.dev_file = None
        self.test_file = './sample/train.txt'

        # preprocessing
        self.norm_digit = True
        # not implemented yet
        self.max_length = 128  # maximum length of input tokens

        # network hyper-prameters
        self.word_rep = 'BiLSTM'
        if self.word_rep == 'BiLSTM':
            # embedding
            self.word_emb_dim = 100
            self.use_char_cnn = False
            if self.use_char_cnn:
                self.char_emb_dim = 10
                self.char_cnn_dim = 30
                self.char_cnn_window_size = 4
                self.max_char_length = 15
            self.lstm_layer = 1
            self.lstm_hidden_dim = 200
            assert(self.lstm_hidden_dim % 2 == 0)
        elif self.word_rep == 'BERT':
            raise NotImplementedError()
        else:
            assert(False)

        self.semi_markov = False
        self.max_NE_length = 1
        if self.semi_markov is False:
            if self.max_NE_length != 1:
                assert(False)

        # training hyper-prameters
        self.epoch = 100
        self.batch_size = 10
        self.dropout_rate = 0.2
        self.grad_algorithm = 'SGD' # or 'Adam'
        self.lr = 0.015
        self.lr_decay = 0.05
        self.momentum = 0.9
        self.weight_decay = float('1e-8')

        self.use_gpu = False

        # PYHSMM
        self.use_PYHSMM = False
        if self.use_PYHSMM:
            if self.semi_markov is False:
                assert(False)
            self.PYHSMM_theta = 2.0
            self.PYHSMM_d = 0.1
            self.PYHSMM_gammaA = 1.0
            self.PYHSMM_gammaB = 1.0
            self.PYHSMM_betaA = 1.0
            self.PYHSMM_betaB = 1.0
            self.PYHSMM_alpha = 1.0
            self.PYHSMM_beta = 1.0
            self.PYHSMM_maxNgram = 2
            self.PYHSMM_maxWordLength = 5
            self.PYHSMM_posSize = -1
            self.PYHSMM_vocabSize = 2097152.0
            self.PYHSMM_epoch = 10
            self.PYHSMM_threads = 8
            self.PYHSMM_batch = 128
            self.PYHSMM_train_file = './sample/train.raw'

        self.save_path = 'save_file_path'  # or None


# return Config instance to main.py
def get_config():
    return Config()
