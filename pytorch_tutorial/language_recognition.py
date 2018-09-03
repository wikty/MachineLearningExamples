import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np



class BoWVocab(object):
    """The Bag-of-Words representation for language sentence.
    
    About BoW representation:
    Each word in the vocab will be assigned an unique index.
    A sentence will be represented by a vector of the count of 
    each word in this sentence.
    """

    def __init__(self, case_sensitivity=False):
        self.index = {}
        self.reverse_index = {}
        self.case_sensitivity = case_sensitivity

    def size(self):
        return len(self.index)

    def word_process(self, word):
        if not self.case_sensitivity:
            return word.lower()
        return word

    def build(self, sentences):
        for sentence in sentences:
            for word in sentence.split():
                word = self.word_process(word)
                i = self.index.setdefault(word, len(self.index))
                self.reverse_index[i] = word
        return len(self.index)

    def dump(self):
        return self.index

    def word2int(self, word):
        word = self.word_process(word)
        return self.index.get(word, None)

    def int2word(self, i):
        return self.reverse_index.get(i, None)

    def sent2vec(self, sentence):
        """Return the BoW representation of a sentence."""
        if isinstance(sentence, str):
            sentence = sentence.split()
        vec = np.zeros(self.size())
        for word in sentence:
            word = self.word_process(word)
            vec[self.word2int(word)] += 1.0
        return torch.from_numpy(vec).float().view(1, -1)


class LangRecognizer(nn.Module):
    """A logistic regression model for the two language 
    recoginition/classification."""

    def __init__(self, vocab_size, num_languages):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_languages)

    def forward(self, x):
        """x is a mini-batch of samples."""
        # Input: the BoW vector with the size of vocab_size
        # Output: the log probability of two languages.
        return F.log_softmax(self.linear(x), dim=1)


def load_data():
    labels = {'ENGLISH': 0, 'SPANISH': 1}
    reverse_labels = {0: 'ENGLISH', 1: 'SPANISH'}

    train_data = np.array([("me gusta comer en la cafeteria", labels['SPANISH']),
        ("Give it to me", labels['ENGLISH']),
        ("No creo que sea una buena idea", labels['SPANISH']),
        ("No it is not a good idea to get lost at sea", labels['ENGLISH'])])

    test_data = np.array([("Yo creo que si", labels['SPANISH']),
                 ("it is lost on me", labels['ENGLISH'])])

    return train_data, test_data, reverse_labels


def get_vocab(train_data, test_data):
    vocab = BoWVocab()
    vocab.build(train_data[:, 0])
    vocab.build(test_data[:, 0])
    return vocab


if __name__ == '__main__':

    train_data, test_data, labels = load_data()

    vocab = get_vocab(train_data, test_data)
    # print(vocab.sent2vec(test_data[0, 0]))

    model = LangRecognizer(vocab.size(), len(labels))
    # Negative log likelihood loss
    # the input to NLLLoss is a vector of log probabilities, and a target label.
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)


    for epoch in range(200):
        print('Epoch: %d' % epoch)
        # train
        for sentence, label in train_data:
            optimizer.zero_grad()
            label = torch.tensor([int(label)])  # a mini-batch
            vec = vocab.sent2vec(sentence)  # a mini-batch
            log_prob = model(vec)
            loss = loss_fn(log_prob, label)
            loss.backward()
            optimizer.step()
        # evaluate
        with torch.no_grad():
            # NOTE: must put the evaluation code block under the no_grad
            for sentence, label in test_data:
                label = int(label)
                vec = vocab.sent2vec(sentence)
                log_prob = model(vec)
                prediction = torch.argmax(log_prob, dim=1).item()
                print('Sentence: %s' % sentence)
                print('Label: {} | Prediction: {} | Correct: {}'.format(
                    labels[label], labels[prediction], (label==prediction)  
                ))
