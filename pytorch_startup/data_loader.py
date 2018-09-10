import os
import random

import torch
import numpy as np

from utils import Params


###
# general purpose
###
class DatasetHandler(object):
    """The handler to read sample of dataset."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __enter__(self):
        # init dataset resources
        return self

    def __exit__(self, type, value, trace):
        # clear dataset resources
        pass

    def read(self):
        # return a sample or None
        pass


class IdentityTransform(object):

    def __init__(self):
        self.name = 'identity'

    def __call__(self, sample):
        return sample


class ComposeTransform(object):

    def __init__(self, transforms=[]):
        self.name = 'compose'
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, sample):
        for transform in self.transforms:
            if sample is None:
                return None
            sample = transform(sample)
        return sample

    def copy(self):
        """Return a copy of this transform."""
        return ComposeTransform(self.transforms)

    def prepend(self, transform):
        """insert `transform` on the first position."""
        self.transform.insert(0, transform)

    def append(self, transform):
        """insert `transform` on the last position."""
        self.transforms.append(transform)

    def insert(self, i, transform):
        """insert `transform` on `i` position."""
        self.transforms.insert(i, transform)


class SplitTransform(object):
    """Split batch into input and output batch."""

    def __call__(self, batch):
        batch_input, batch_output = [], []
        for sample in batch:
            batch_input.append(sample[0])
            batch_output.append(sample[1])
        return batch_input, batch_output


class ToArrayTransform(object):
    """Convert list to numpy array."""

    def __init__(self, dtype=None):
        """
        Args:
            dtype (np.dtype): numpy dtype.
        """
        self.dtype = dtype

    def __call__(self, batch):
        batch_input, batch_target = batch
        return (np.array(batch_input, dtype=self.dtype), 
            np.array(batch_target, dtype=self.dtype))


class ToTensorTransform(object):

    def __init__(self, dtype=None, to_cuda=None):
        """
        Args:
            dtype (torch.dtype): torch dtype.
            to_cude (bool|device): flag to enable cude or the destination
                GPU device instance.
        """
        self.dtype = dtype
        self.to_cuda = to_cuda

    def __call__(self, batch):
        batch_input, batch_target = batch
        if not self.to_cuda:
            return (torch.tensor(batch_input, dtype=self.dtype), 
                torch.tensor(batch_target, dtype=self.dtype))
        elif isinstance(self.to_cuda, bool):
            return (torch.tensor(batch_input, dtype=self.dtype).cuda(),
                torch.tensor(batch_target, dtype=self.dtype).cuda())
        else:
            return (
                torch.tensor(batch_input, 
                             dtype=self.dtype).cuda(self.to_cuda),
                torch.tensor(batch_target, 
                             dtype=self.dtype).cuda(self.to_cuda)
            )


class Dataset(object):
    """Iterator for dataset."""

    def __init__(self, name, size, handler, transform=None):
        self.name = name
        self.size = size
        self.handler = handler
        self.transform = (lambda s: s) if transform is None else transform

    def __iter__(self):
        """Return a generator that returns a sample in each iteration."""
        with self.handler as handler:
            while True:
                sample = handler.read()
                if sample is None:
                    break
                sample = self.transform(sample)
                yield sample


class BatchGenerator(object):

    def __init__(self, dataset, batch_size=1, shuffle=False, transform=None):
        self.dataset = dataset
        self.size = batch_size
        self.shuffle = shuffle
        self.transform = (lambda s: s) if transform is None else transform

    @property
    def name(self):
        """The name of dataset."""
        return self.dataset.name

    @property
    def dataset_size(self):
        """The size of dataset."""
        return self.dataset.size
    
    @property
    def batch_size(self):
        """The size of batch."""
        return self.size

    def __iter__(self):
        batch = []
        dataset = self.dataset
        if self.shuffle:
            dataset = list(dataset)  # can't benefit from lazy loading
            random.shuffle(dataset)  # shuffle dataset in-place
        for sample in dataset:
            batch.append(sample)
            if len(batch) == self.size:
                yield self.transform(batch)
                batch = []
        if batch:
            # the last batch may be less then batch size.
            yield self.transform(batch)


###
# specific project
###
class Vocabulary(object):
    """Vocab maps a string to integer or one-hot vector.
    
    Note: The index of vocab start from zero, and auto-increased when new
    term is added into vocab.
    """

    def __init__(self, raw_txt=None, encoding='utf8', 
                 one_hot=False, case_sensitive=True):
        self.one_hot = one_hot
        self.case_sensitive = case_sensitive
        self.terms = {}
        self.rterms = {}
        if raw_txt:
            self.build(raw_txt, encoding)

    def __len__(self):
        return len(self.terms)

    def size(self):
        return len(self)

    def process(self, term):
        """The term processor."""
        term = str(term)
        if not self.case_sensitive:
            term = term.lower()
        return term

    def clear(self):
        """Clear vocab.
        
        Returns: return self.
        """
        self.terms = {}
        self.rterms = {}
        return self

    def build(self, raw_txt, encoding='utf8'):
        """Build a new vocab.

        Args:
            raw_txt (path): txt file, the terms split by space and end-of-line

        Returns: return self
        """
        assert os.path.isfile(raw_txt)
        with open(raw_txt, 'r', encoding=encoding) as f:
            for term in f.read().split():
                self.add(term)
        return self

    def add(self, term):
        """
        Args:
            term (str): the term string.

        Return: return self
        """
        term = self.process(term)
        i = self.terms.setdefault(term, len(self.terms))
        self.rterms[i] = term
        return self

    def encode(self, term, default=None):
        """Term to integer or one-hot encoding."""
        term = self.process(term)
        i = self.terms.get(term, default)
        if i == default:
            return default
        if self.one_hot:
            a = np.zeros(self.size())
            a[i] = 1.0
            return a
        return i

    def decode(self, i, default=None):
        """Integer or one-hot encoding to the term."""
        if self.one_hot:
            i = np.argmax(i)
        return self.rterms.get(i, default)


class NERHandler(DatasetHandler):

    def __init__(self, input_file, target_file, encoding='utf8'):
        self.input_file = input_file
        self.target_file = target_file
        self.encoding = encoding
        self.input_fh = None
        self.target_fh = None

    def __enter__(self):
        self.input_fh = open(self.input_file, 'r', encoding=self.encoding)
        self.target_fh = open(self.target_file, 'r', encoding=self.encoding)
        return self

    def __exit__(self, type, value, trace):
        self.input_fh.close()
        self.target_fh.close()
        self.input_fh = None
        self.target_fh = None

    def read(self):
        """Return a sample or None"""
        # handler not open
        if self.input_fh is None or self.target_fh is None:
            return None
        # each line is a sample
        input_line = self.input_fh.readline()
        target_line = self.target_fh.readline()
        # inputs and targets must be with the same size
        assert ((len(input_line) > 0 and len(target_line) > 0) or 
            (len(input_line) == 0 and len(target_line) == 0))
        if len(input_line) == 0:
            return None
        # split only by space ASCII 32, not include ASCII 160
        tokens = input_line.rstrip('\r\n').split(' ')
        labels = target_line.rstrip('\r\n').split(' ')
        msg = "tokens[{}] and labels[{}] of sentence {} don't match in file {}"
        assert len(tokens) == len(labels), \
            msg.format(len(tokens), len(labels), input_line, self.input_file)
        return (tokens, labels)


class VocabTransform(object):
    """A transform for sample."""

    def __init__(self, word_vocab, tag_vocab, unk_word):
        self.unk_word = unk_word
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab

    def __call__(self, sample):
        if sample is None:
            return None
        tokens, labels = sample
        tokens_encoding, labels_encoding = [], []
        for token in tokens:
            encoding = self.word_vocab.encode(token, default=None)
            if encoding is None:
                encoding = self.word_vocab.encode(self.unk_word)
                assert encoding is not None  # unknow word must be in vocab
            tokens_encoding.append(encoding)
        for label in labels:
            encoding = self.tag_vocab.encode(label, default=None)
            assert encoding is not None  # tag must be in vocab
            labels_encoding.append(encoding)
        return tokens_encoding, labels_encoding


class PaddingTransform(object):
    """A transform for mini-batch."""

    def __init__(self, word_padding, tag_padding):
        self.word_padding = word_padding
        self.tag_padding = tag_padding

    def __call__(self, batch):
        sample_max_len = 0
        for sample in batch:
            tokens, labels = sample
            sample_len = len(tokens)
            if sample_len > sample_max_len:
                sample_max_len = sample_len
        new_batch = []
        for sample in batch:
            tokens, labels = sample
            assert len(tokens) == len(labels)
            padding_len = sample_max_len - len(tokens)
            tokens.extend([self.word_padding]*padding_len)
            labels.extend([self.tag_padding]*padding_len)
            new_batch.append((tokens, labels))
        return new_batch


class DataLoader(object):

    def __init__(self, data_dir, params, encoding='utf8'):
        assert os.path.isdir(data_dir)
        assert isinstance(params, Params)
        # info about dataset
        self.word_vocab_filename = 'words.txt'
        self.tag_vocab_filename = 'tags.txt'
        self.sentences_filename = 'sentences.txt'
        self.labels_filename = 'labels.txt'
        self.data_dir = data_dir
        # load dataset params
        self.params = params.copy()
        self.data_size = {
            'train': self.params.train_size,
            'val': self.params.val_size,
            'test': self.params.test_size
        }
        # sample transform
        words_file=os.path.join(self.data_dir, self.word_vocab_filename)
        tags_file=os.path.join(self.data_dir, self.tag_vocab_filename)
        word_vocab = Vocabulary(words_file, encoding=encoding)
        tag_vocab = Vocabulary(tags_file, encoding=encoding)
        self.sample_transform = VocabTransform(
            word_vocab=word_vocab,
            tag_vocab=tag_vocab,    
            unk_word=self.params.unk_word
        )
        # batch transform
        word_padding = word_vocab.encode(self.params.pad_word)
        tag_padding = -1  # don't use self.params.pad_tag
        self.batch_transform = ComposeTransform([
            PaddingTransform(word_padding, tag_padding),
            SplitTransform(), 
            ToArrayTransform()
        ])

    def parameters(self):
        """A copy of dataset parameters."""
        return self.params.copy()

    def load(self, name, encoding='utf8', batch_size=1, shuffle=False, 
             to_tensor=True, to_cuda=None):
        """
        Args:
            to_tensor (bool): enable numpy array to torch tensor.
            to_cuda (bool|device): `True` means use the current GPU device.
                `Flase` or `None` means use CPU device. `device` specifies 
                the destination GPU device.
        """
        dataset_dir = os.path.join(self.data_dir, name)
        msg = 'dataset directory {} not exists'.format(dataset_dir)
        assert os.path.isdir(dataset_dir), msg
        # dataset handler to read sample
        sentences_file = os.path.join(dataset_dir, self.sentences_filename)
        labels_file = os.path.join(dataset_dir, self.labels_filename)
        handler = NERHandler(
            input_file=sentences_file,
            target_file=labels_file,
            encoding=encoding
        )
        # create dataset
        dataset = Dataset(name, self.data_size[name], handler, 
                          self.sample_transform)
        # update batch transform
        batch_transform = self.batch_transform.copy()
        if to_tensor:
            # Note: model's word-embedding requires `LongTensor`
            batch_transform.append(ToTensorTransform(dtype=torch.long,
                                                     to_cuda=to_cuda))
        # create batch generator
        generator = BatchGenerator(dataset, 
                                   batch_size=batch_size, 
                                   shuffle=shuffle, 
                                   transform=batch_transform)
        return generator
