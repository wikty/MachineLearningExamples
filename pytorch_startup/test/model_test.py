import unittest

import config
from model import Net, criterion, accuracy
from data_loader import DataLoader
from utils import Params, RunningAvg, Logger


class ModelTest(unittest.TestCase):

    def setUp(self):
        self.trainset = config.train_name
        self.valset = config.val_name
        self.testset = config.test_name
        self.tag_padding = -1
        self.params = Params(config.datasets_params_file)
        self.params.embedding_dim = 50
        self.params.lstm_hidden_dim = 50
        self.loader = DataLoader(config.data_dir, self.params)
        self.logger = Logger.get()

    def test_forward_and_backward(self):
        # Note: model and dataset all transfer to CUDA device.
        batches = self.loader.load(self.trainset, batch_size=10, 
                                   shuffle=True, to_tensor=True, 
                                   to_cuda=True)
        model = Net(
            word_vocab_size=self.params.word_vocab_size,
            tag_vocab_size=self.params.tag_vocab_size,
            embedding_dim=self.params.embedding_dim,
            lstm_hidden_dim=self.params.lstm_hidden_dim
        ).cuda()
        for batch in batches:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.logger.debug('loss: {}'.format(loss.item()))
            break

    def test_accuracy(self):
        batches = self.loader.load(self.valset, batch_size=10,
                                   shuffle=True, to_tensor=True)
        model = Net(
            word_vocab_size=self.params.word_vocab_size,
            tag_vocab_size=self.params.tag_vocab_size,
            embedding_dim=self.params.embedding_dim,
            lstm_hidden_dim=self.params.lstm_hidden_dim
        )
        running_avg = RunningAvg()
        for batch in batches:
            inputs, targets = batch
            outputs = model(inputs)  # should be used with no_grad()
            acc = accuracy(outputs.data, targets.data)
            running_avg.step(acc.item())
        self.logger.debug('running accuracy: {}'.format(running_avg()))
