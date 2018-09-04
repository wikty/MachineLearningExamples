import torch
from torch import nn
import torch.optim as optim
from torch.nn import Embedding
import torch.nn.functional as F


# Words are common features in the natural language process.
# How can we represent words?
# 1. ASCII encoding. What about the meaning of words?
# 2. One-hot encoding. It's too sparse to represent the whole language vocab.
#    And one-hot encoding cannot represent the similarity between words.
# 3. Word-Embedding. a dense encoding way that can represent semantic 
#    similarity between words.


###
# one-hot encoding for word
###
class OneHotEncoding(object):
    """The sparse representation of word.
    
    There is an enormous drawback to this representation, besides just how
    huge it is. It basically treats all words as independent entities with
    no relation to each other. What we really want is some notion of 
    similarity between words.
    """

    def __init__(self):
        self._index = {}
        self._rindex = {}

    def build(self, sentences):
        for sentence in sentences:
            for word in sentence.split():
                i = self._index.setdefault(word, len(self._index))
                self._rindex[i] = word
        return len(self._index)

    def vocab_size(self):
        return len(self._index)

    def encode(self, word):
        i = self._index.get(word, None)
        vec = None
        if i is not None:
            vec = [0] * self.vocab_size()
            vec[i] = 1
        return vec

    def decode(self, vec):
        i = vec.index(1)
        return self._rindex.get(i, None)


def play_onehot():
    data = ['hello world']
    onehot = OneHotEncoding()
    onehot.build(data)
    encoding = onehot.encode('hello')
    print(encoding)
    print(onehot.decode(encoding))


###
# word-embedding representation for word
###
class WordEmbedding(object):
    """word embeddings are a dense representation of the semantics of a word.

    Motivation:
    It is a technique to combat the sparsity of linguistic data, by connecting
    the dots between what we have seen and what we haven’t. It relies on a 
    fundamental linguistic assumption: that words appearing in similar contexts
    are related to each other semantically. This is called the distributional 
    hypothesis.

    Definition
    Each word is encoded as a vector of many semantic attributes. So the 
    entries in the vector are mostly non-zero.
    Then we can get a measure of similarity between these words by doing 
    vector dot-product. 
    You can think of the sparse one-hot vectors as a special case of word 
    embeddings, where each word basically has similarity 0, and we gave each 
    word some unique semantic attribute.

    Representation:
    Which of semantic attributes should we use to represent words, and how 
    can we set the value of thos attributes for words?
    We'll use neural network to learn those semantic features. More precise,
    those features are latent semantic attributes, they are not clear what 
    that means, i.e., the word embeddings will probably not be interpretable
    to us.    
    """
    def __init__(self, embed_size):
        self._index = {}  # word to index
        self._rindex = {}  # index to word
        self._embeddings = None  # index to embeding representation
        self.embed_size = embed_size

    def build(self, sentences):
        for sentence in sentences:
            for word in sentence.split():
                i = self._index.setdefault(word, len(self._index))
                self._rindex[i] = word
        vocab_size = len(self._index)
        # NOTE: `nn.Embedding` is just a container of parameters.
        self._embeddings = Embedding(vocab_size, self.embed_size)
        return vocab_size

    def lookup(self, word):
        """Return a mini-batch of the word representation."""
        i = self.index(word)
        if i is None:
            return None
        t = torch.tensor([i], dtype=torch.long)  # must be long int
        return self._embeddings(t)

    def vocab_size(self):
        return len(self._index)

    def index(self, word):
        return self._index.get(word, None)

    def rindex(self, i):
        return self._rindex.get(word, None)


def play_word_embedding():
    data = ['hello world']
    embed = WordEmbedding(embed_size=20)
    embed.build(data)
    print(embed.lookup('hello'))


###
# N-Gram language model learning/word-embedding learning
###
class NGramLanguageModel(nn.Module):
    """N-Gram language model.

    In an n-gram language model, given a sequence of words, we want to
    compute: 

    $$P(w_i | w_{i-1}, w_{i-2}, \dots, w_{i-n+1} )$$

    We want to train a n-gram language model and learn the word-embeddings
    representation.
    """

    def __init__(self, vocab_size, embedding_size, context_size):
        super().__init__()
        # nn.Embedding is a container of parameters
        self.embeding = Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(context_size*embedding_size, 120)
        self.linear2 = nn.Linear(120, vocab_size)

    def forward(self, x):
        """Return the probability distribution of the next word that follow 
        the sequence of input words.
        
        Note: the input `x` is a sequence of words indices.
        """
        v = self.embeding(x).view(1, -1)
        h = F.relu(self.linear1(v))
        log_prob = F.log_softmax(self.linear2(h), dim=1)
        return log_prob

    def predict(self, x):
        with torch.no_grad():
            log_prob = self.forward(x)
        return torch.argmax(log_prob, dim=1)

    def embed(self, x):
        """Return the learned word-embedding vector."""
        v = None
        with torch.no_grad():
            v = self.embeding(x).view(1, -1)
        return v


###
# Continuous Bag-of-Words model training/word-embedding pretraining
###
class CBOWModel(nn.Module):
    """Continuous Bag-of-Words model.

    It is a model that tries to predict words given the context of 
    a few words before and a few words after the target word. 
    
    This is distinct from language modeling, since CBOW is not 
    sequential and does not have to be probabilistic. 
    
    Typcially, CBOW is used to quickly train word embeddings, and
    these embeddings are used to initialize the embeddings of some
    more complicated model. 
    """
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        # nn.Embedding is a container of parameters
        self.embeding = Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def forward(self, x):
        """`x` is the context words indices."""
        v = self.embeding(x)
        # sum each context word's embedding row vector
        v = torch.sum(v, dim=0).view(1, -1)
        # affine map
        h = self.linear(v)
        log_prob = F.log_softmax(h, dim=1)
        return log_prob

    def predict(self, x):
        with torch.no_grad():
            log_prob = self.forward(x)
        return torch.argmax(log_prob, dim=1)

    def embed(self, x):
        """Return the learned word-embedding vector."""
        v = None
        with torch.no_grad():
            v = self.embeding(x).view(1, -1)
        return v


class WordIndex(object):
    def __init__(self):
        self._index = {}  # word to index
        self._rindex = {}  # index to word

    def build(self, sentences):
        for sentence in sentences:
            for word in sentence.split():
                i = self._index.setdefault(word, len(self._index))
                self._rindex[i] = word
        return len(self._index)

    def vocab_size(self):
        return len(self._index)

    def index(self, word, to_tensor=False):
        i = self._index.get(word, None)
        if i is None:
            return None
        if not to_tensor:
            return i
        return torch.tensor([i], dtype=torch.long)  # long int with 1-dim

    def rindex(self, i):
        return self._rindex.get(i, None)


def load_data_ngram():
    text = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold."""
    idx = WordIndex()
    idx.build([text])
    words = text.split()
    # 3-gram: each item is a tuple([w_i-2, w_i-1], w_i)
    trigrams = [
        ([words[i-2], words[i-1]], words[i]) for i in range(2, len(words))
    ]
    return idx, trigrams, 2


def load_data_cbow():
    text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells."""
    idx = WordIndex()
    idx.build([text])
    words = text.split()
    # context size is 2: each item is a tuple of
    # ((w_i-2, w_i-1, w_i+1, w_i+2), w_i)
    data = [
        ((words[i-2], words[i-1], words[i+1], words[i+2]), words[i])
        for i in range(2, len(words)-2)
    ]
    return idx, data, 2


def train_ngram_language_model(epoch=500, embedding_size=10, 
    learning_rate=0.001):
    idx, trigrams, context_size = load_data_ngram()
    model = NGramLanguageModel(idx.vocab_size(), 
                               embedding_size, context_size)
    return idx, train_model(model, trigrams, idx,
                       epoch, learning_rate)


def train_cbow_model(epoch=500, embedding_size=10,
    learning_rate=0.001):
    idx, dataset, context_size = load_data_cbow()
    model = CBOWModel(idx.vocab_size(), embedding_size)
    return idx, train_model(model, dataset, idx,
                       epoch, learning_rate)


def train_model(model, dataset, idx, epoch=500, learning_rate=0.001):
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for e in range(epoch):
        # train
        for context_words, target_word in dataset:
            optimizer.zero_grad()
            log_prob = model(torch.cat([
                idx.index(w, True) for w in context_words
            ]))
            loss = loss_fn(log_prob, idx.index(target_word, True))
            loss.backward()
            optimizer.step()
        # evaluate
        corrects = 0
        with torch.no_grad():
            for context_words, target_word in dataset:
                target = idx.index(target_word, True)
                predict = model.predict(torch.cat([
                    idx.index(w, True) for w in context_words
                ]))
                if target.item() == predict.item():
                    corrects = corrects + 1
        print('Epoch: {}, Acc: {}%%'.format(
            e, 100*corrects/len(dataset)
        ))
    return model



def ngram_predict(model, idx):
    data = [
        (('within', 'thine'), 'own'),
        (('were', 'to'), 'be'),
        (('Proving', 'his'), 'beauty')
    ]
    with torch.no_grad():
        for context_words, target_word in data:
            predict = model.predict(torch.cat([
                idx.index(w, True) for w in context_words
            ]))
            print('Context:', context_words)
            print('Target: %s' % target_word)
            print('Prediction: %s' % idx.rindex(predict.item()))


def ngram_embedding(model, idx):
    with torch.no_grad():
        v1 = model.embed(idx.index('when', True))
        v2 = model.embed(idx.index('where', True))
        v3 = model.embed(idx.index('Where', True))

        print('when and where / similarity: {}'.format(
            F.cosine_similarity(v1, v2)
        ))
        print('where and Where / similarity: {}'.format(
            F.cosine_similarity(v2, v3)
        ))


def cbow_embedding(model, idx):
    with torch.no_grad():
        v1 = model.embed(idx.index('are', True))
        v2 = model.embed(idx.index('is', True))
        print('are and is / similarity: {}'.format(
            F.cosine_similarity(v1, v2)
        ))

        v3 = model.embed(idx.index('study', True))
        v4 = model.embed(idx.index('pattern', True))
        print('study and pattern / similarity: {}'.format(
            F.cosine_similarity(v3, v4)
        ))


if __name__ == '__main__':
    # play_onehot()
    # play_word_embedding()
    
    # idx, model = train_ngram_language_model()
    # ngram_predict(model, idx)
    # ngram_embedding(model, idx)
    
    idx, model = train_cbow_model()
    cbow_embedding(model, idx)