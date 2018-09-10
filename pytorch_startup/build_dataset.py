import os
import csv
import json
import random
import argparse
import collections

import config
from utils import Logger, Params


###
# general purpose
###
class Sample(object):
    
    def __init__(self, **kwargs):
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
        self.__dict__.update(kwargs)

    def __getattr__(self, key):
        """Raise exception when the key not exists in the instance."""
        raise AttributeError("{} not exists in sample.".format(key))


class Dataset(object):

    def __init__(self, name, samples=[]):
        self._name = name
        self._samples = [sample for sample in samples]

    def __len__(self):
        return len(self._samples)

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return len(self._samples)

    def add(self, sample):
        self._samples.append(sample)

    def extend(self, samples=[]):
        self._samples.extend(samples)

    def get(self, shuffle=False, key=None, **key_kwargs):
        """Return a iterator of the dataset.
        
        Args:
            shuffle (bool): shuffle the dataset
            key (function): a filter function for dataset
            key_kwargs (dict): arguments for the key filter function
        """
        foo = list(range(len(self._samples)))
        if shuffle:
            random.shuffle(foo)
        for i in foo:
            sample = self._samples[i]
            if key is not None and (not key(sample, **key_kwargs)):
                continue
            yield sample

    def __iter__(self):
        return self.get()


###
# specific project
###
class VocabCounter(object):
    """A vocab counter for words and tags."""

    def __init__(self, items=[], min_count=1, max_count=None):
        self.counter = collections.Counter()
        self.min_count = min_count
        self.max_count = max_count
        self.update(items)

    def __len__(self):
        return len(self.counter)

    def update(self, items=[]):
        self.counter.update(items)

    def reset(self):
        self.counter.clear()

    def get(self, min_count=None, max_count=None):
        """Return the items whose count in [min_count, max_count]."""
        min_count = self.min_count if min_count is None else min_count
        max_count = self.max_count if max_count is None else max_count
        s = set()
        for key, count in self.counter.items():
            if min_count is not None and count < min_count:
                continue
            if max_count is not None and count > max_count:
                continue
            s.add(key)
        return s


class Builder(object):

    # parameters for your project
    PAD_WORD = '<pad>'  # pad for word in vocab
    UNK_WORD = 'UNK'  # unknow word in vocab
    PAD_TAG = 'O'  # pad for tag in vocab
    sentences_filename = 'sentences.txt'
    labels_filename = 'labels.txt'
    word_vocab_filename = 'words.txt'
    tag_vocab_filename = 'tags.txt'

    def __init__(self, datasets_params_file, train_factor=0.7, 
                 val_factor=0.15, test_factor=0.15, train_dirname='train', 
                 val_dirname='val', test_dirname='test'):
        self.datasets_params_file = datasets_params_file
        self.train_factor = train_factor
        self.val_factor = val_factor
        self.test_factor = test_factor
        self.train_dirname = train_dirname.strip('/').strip('\\')
        self.val_dirname = val_dirname.strip('/').strip('\\')
        self.test_dirname = test_dirname.strip('/').strip('\\')
        self.samples = []
        self.logger = Logger.get()

    def datasets(self, shuffle=True):
        if shuffle:
            random.shuffle(self.samples)
        l = len(self.samples)
        i = int(l*self.train_factor)
        j = int(l*(self.train_factor+self.val_factor))
        return (
            Dataset(self.train_dirname, self.samples[:i]),
            Dataset(self.val_dirname, self.samples[i:j]),
            Dataset(self.test_dirname, self.samples[j:])
        )

    def load(self, csv_file, encoding='utf8'):
        """Do dirty job, you should modify it to suit for your project."""
        self.logger.info('Loading dataset from csv file...')
        with open(csv_file, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f)
            words, tags = [], []
            for row in reader:
                if row['Sentence #'].strip() and len(words):
                    assert len(words) == len(tags)
                    self.samples.append(Sample(words=words, tags=tags))
                    words, tags = [], []
                try:
                    word, tag = str(row['Word']), str(row['Tag'])
                except UnicodeDecodeError as e:
                    msg = "An exception was raised, skipping a word: {}"
                    self.logger.warning(msg.format(e))
                    pass
                else:
                    words.append(word)
                    tags.append(tag)
            if len(words) > 0:
                assert len(words) == len(tags)
                self.samples.append(Sample(words=words, tags=tags))
        self.logger.info('The number of samples is {}'.format(
            len(self.samples)))
        self.logger.info('- done!')

    def dump(self, data_dir, encoding='utf8', shuffle=True,
             min_count_word=1, min_count_tag=1):
        """Do dirty job, you should modify it to suit for your project."""        
        # datasets params
        params = Params(data={
            'word_vocab_size': 0,
            'tag_vocab_size': 0,
            'pad_word': self.PAD_WORD,
            'unk_word': self.UNK_WORD,
            'pad_tag': self.PAD_TAG
        })
        # dataset and vocab
        tag_vocab = VocabCounter([self.PAD_TAG])
        word_vocab = VocabCounter([self.PAD_WORD, self.UNK_WORD])
        datasets = self.datasets(shuffle=shuffle)
        # save train/val/test dataset
        for dataset in datasets:
            name = dataset.name
            size = len(dataset)
            self.logger.info('Saving {} dataset...'.format(name))
            params.set('{}_size'.format(name), size)  # add dataset size
            dirpath = os.path.join(data_dir, name)
            if not os.path.isdir(dirpath):
                os.makedirs(dirpath)
            sentences_file = os.path.join(dirpath, self.sentences_filename)
            labels_file = os.path.join(dirpath, self.labels_filename)
            with open(sentences_file, 'w', encoding=encoding) as fs, \
                open(labels_file, 'w', encoding=encoding) as fl:
                for sample in dataset:
                    words, tags = sample.words, sample.tags
                    fs.write('{}\n'.format(' '.join(words)))
                    fl.write('{}\n'.format(' '.join(tags)))
                    tag_vocab.update(tags)
                    word_vocab.update(words)            
            self.logger.info('- done!')
        params.word_vocab_size = len(word_vocab)
        params.tag_vocab_size = len(tag_vocab)
        # save word vocab
        self.logger.info('Saving word vocab...')
        word_vocab_file = os.path.join(data_dir, 
                                       self.word_vocab_filename)
        with open(word_vocab_file, 'w', encoding=encoding) as f:
            for word in word_vocab.get(min_count=min_count_word):
                f.write('{}\n'.format(word))
        self.logger.info('- done!')
        # save tag vocab
        self.logger.info('Saving tag vocab...')
        tag_vocab_file = os.path.join(data_dir, 
                                      self.tag_vocab_filename)
        with open(tag_vocab_file, 'w', encoding=encoding) as f:
            for tag in tag_vocab.get(min_count=min_count_tag):
                f.write('{}\n'.format(tag))
        self.logger.info('- done!')
        # save datasets parameters
        self.logger.info('Saving datasets parameters...')
        params.dump(self.datasets_params_file, encoding=encoding)
        self.logger.info('- done!')
        # print dataset characteristics
        self.logger.info("Characteristics of the dataset:")
        for key, value in params:
            self.logger.info("- {}: {}".format(key, value))


if __name__ == '__main__':
    # default settings
    data_dir = config.data_dir
    datasets_log = config.datasets_log
    datasets_params_file = config.datasets_params_file
    data_file = os.path.join(data_dir, 'ner_dataset.csv')
    min_count_word, min_count_tag = 1, 1
    train_factor, val_factor, test_factor = (config.train_factor,
                                             config.val_factor,
                                             config.test_factor)
    train_dirname, val_dirname, test_dirname = (config.train_name,
                                                config.val_name,
                                                config.test_name)

    def tofloat(x):
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in [0.0, 1.0]" % x)
        return x

    def toint(x):
        x = int(x)
        if x < 0:
            raise argparse.ArgumentTypeError("%r must be greater then 0" % x)
        return x

    # define arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', 
        default=data_file,
        help="File of data source")
    parser.add_argument('--data-dir', 
        default=data_dir,
        help="Directory for the dataset")
    parser.add_argument('--train-factor', 
        default=train_factor,
        help="The factor of train dataset", 
        type=tofloat)
    parser.add_argument('--val-factor', 
        default=val_factor,
        help="The factor of validation dataset", 
        type=tofloat)
    parser.add_argument('--test-factor', 
        default=test_factor,
        help="The factor of test dataset", 
        type=tofloat)
    parser.add_argument('--min-count-word', 
        default=min_count_word, 
        help="Minimum count for words in the dataset(default: %(default)s)", 
        type=toint)
    parser.add_argument('--min-count-tag', 
        default=min_count_tag, 
        help="Minimum count for tags in the dataset(default: %(default)s)", 
        type=toint)

    # parse command line arguments
    args = parser.parse_args()

    # get and check arguments
    data_file, data_dir = args.data_file, args.data_dir
    train_factor = args.train_factor
    val_factor = args.val_factor
    test_factor = args.test_factor
    min_count_word = args.min_count_word
    min_count_tag = args.min_count_tag
    msg = '{} file not found. Make sure you have downloaded it.'
    assert os.path.isfile(data_file), msg.format(data_file)
    msg = '{} directory not found. Please create it first.'
    assert os.path.isdir(data_dir), msg.format(data_dir)
    msg = 'train factor + val factor + test factor must be equal to 1.0'
    assert (1.0 == (train_factor + val_factor + test_factor)), msg

    # set logger
    Logger.set(datasets_log)

    builder = Builder(datasets_params_file,
                      train_factor=train_factor, 
                      val_factor=val_factor,
                      test_factor=test_factor,
                      train_dirname=train_dirname,
                      val_dirname=val_dirname,
                      test_dirname=test_dirname)
    builder.load(data_file, 
                 encoding='windows-1252')
    builder.dump(data_dir, 
                 encoding='utf8',
                 min_count_word=min_count_word,
                 min_count_tag=min_count_tag)


