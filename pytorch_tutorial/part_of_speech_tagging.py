import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)


class LSTMTagger(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, tag_size):
        super().__init__()
        # sentence(word indices) to word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # word embeddings to hidden states
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # hidden states to the tag space
        self.h2t = nn.Linear(hidden_dim, tag_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.hidden_dim = hidden_dim
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # (num_layers, minibatch_size, hidden_dim)
        return (torch.randn(1, 1, self.hidden_dim),
                torch.randn(1, 1, self.hidden_dim))

    def forward(self, sentence):
        """
        Params:
            sentence: a list of word index.
        """
        embeddings = self.embedding(sentence)
        # Add batch dim before pass into LSTM.
        # because the sentence doesn't have batch dim, but LSTM layer 
        # requires that.
        embeddings = embeddings.view(len(sentence), 1, -1)
        # The first value returned by LSTM is all of the hidden states
        # throughout the sequence. The second is just the most recent 
        # hidden state.
        outputs, self.hidden = self.lstm(embeddings, self.hidden)
        outputs = outputs.view(len(sentence), -1)  # remove batch dim
        outputs = self.h2t(outputs)
        return self.softmax(outputs)


def load_data():
    return [
        ("The dog ate the apple".split(), 
            ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(),
            ["NN", "V", "DET", "NN"])
    ]


def build_idx(samples):
    word_idx, tag_idx = {}, {}
    for words, tags in samples:
        for word in words:
            if word not in word_idx:
                word_idx[word] = len(word_idx)
        for tag in tags:
            if tag not in tag_idx:
                tag_idx[tag] = len(tag_idx)
    return word_idx, tag_idx


def token2tensor(idx, tokens):
    indices = [idx[t] for t in tokens]
    return torch.tensor(indices, dtype=torch.long)


if __name__ == '__main__':
    samples = load_data()
    word_idx, tag_idx = build_idx(samples)
    tag_ridx = {v:k for k, v in tag_idx.items()}

    num_epochs = 1000
    learning_rate = 0.01
    vocab_size = len(word_idx)
    embedding_dim = 128
    hidden_dim = 128
    tag_size = len(tag_idx)

    tagger = LSTMTagger(vocab_size, embedding_dim, hidden_dim, tag_size)
    criterion = F.nll_loss
    optimizer = optim.SGD(tagger.parameters(), lr=learning_rate)

    # Train
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        avg_loss = 0.0
        for words, tags in samples:
            words = token2tensor(word_idx, words)
            tags = token2tensor(tag_idx, tags)

            # reset init hidden
            tagger.hidden = tagger.init_hidden()

            # forward
            outputs = tagger(words)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

        print('Loss {}'.format(avg_loss/len(samples)))

    # Predict
    with torch.no_grad():
        for words, tags in samples:
            words_tensor = token2tensor(word_idx, words)
            outputs = tagger(words_tensor)
            indices = torch.argmax(outputs, dim=1).squeeze()
            pred_tags = [tag_ridx[i.item()] for i in indices]
            print('Sentence:', ' '.join(words))
            print('Pred:   ', ' '.join(pred_tags))
            print('Tags:   ', ' '.join(tags))