import os
import json
import math
import shutil
import copy
import logging

import torch


def load_from_json(json_file, encoding='utf8'):
    with open(json_file, 'r', encoding=encoding) as f:
        return json.load(f)


def dump_to_json(data, json_file, encoding='utf8', indent=4):
    with open(json_file, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


class Params(object):
    """Load/dump parameters from/to json file."""

    def __init__(self, json_path=None, encoding='utf8', data={}):
        if json_path:
            self.load(json_path, encoding)
        if data and isinstance(data, dict):
            self.__dict__.update(data)

    def __getattr__(self, key):
        """Raise exception when the key not exists in the instance."""
        raise AttributeError("{} not exists in params.".format(key))

    def __iter__(self):
        for key, value in self.__dict__.items():
            yield key, value

    @property
    def dict(self):
        """A copy with dict-like interface to Params instance."""
        return copy.deepcopy(self.__dict__)

    def items(self):
        """A interface like dict.items()."""
        return self.__dict__.items()

    def load(self, json_path, encoding='utf8'):
        """Load and update parameters from json file."""
        with open(json_path, 'r', encoding=encoding) as f:
            data = json.load(f)
            self.__dict__.update(data)

    def set(self, name, value):
        """Set a attribute for params."""
        setattr(self, name, value)

    def update(self, params):
        """Update parameters from another Params instance."""
        assert isinstance(params, Params)
        self.__dict__.update(params.dict)

    def dump(self, json_path, encoding='utf8', indent=4):
        """Dump parameters to json file."""
        with open(json_path, 'w', encoding=encoding) as f:
            json.dump(self.__dict__, f, 
                      ensure_ascii=False, indent=indent)

    def copy(self):
        params = Params()
        params.update(self)
        return params


class RunningAvg():
    """Running average of a quantity or a list/tuple/dict of quantities.

    Example:
    ```
    # average for numbers
    loss_avg = RunningAvg()
    loss_avg.step(2)
    loss_avg.step(4)
    loss_avg()  # return 3.0

    # average for a dict of numbers
    loss_avg = RunningAvg()
    loss_avg.step({'loss': 1, 'accuracy': 2})
    loss_avg.update({'loss': 0, 'accuracy': 4})
    loss_avg()  # return {'loss': 0.5, 'accuracy': 3.0}
    ```
    """

    def __init__(self):
        self.data = None
        self.steps = 0

    def _v(self, val):
        try:
            val = float(val)
        except Exception as e:
            msg = 'Running average quantity must be able to convert to float.'
            raise Exception(msg)
        return val

    def process(self, val):
        assert isinstance(val, (int, float, list, tuple, dict))
        if isinstance(val, (int, float)):
            val = self._v(val)
        elif isinstance(val, (tuple, list)):
            val = [self._v(v) for v in val]
        elif isinstance(val, dict):
            val = {k:self._v(v) for k, v in val.items()}
        return val

    def reset(self):
        self.data = None
        self.steps = 0

    def step(self, val):
        if self.data is None:
            self.data = self.process(val)
        elif isinstance(self.data, float):
            assert isinstance(val, (int, float))
            self.data += self.process(val)
        elif isinstance(self.data, list):
            assert isinstance(val, (tuple, list))
            assert len(val) == len(self.data)
            val = self.process(val)
            for i in range(len(val)):
                self.data[i] += val[i]
        elif isinstance(self.data, dict):
            assert isinstance(val, dict)
            assert len(val) == len(self.data)
            val = self.process(val)
            for key in val:
                self.data[key] += val[key]
        self.steps += 1

    def __call__(self):
        if self.steps == 0:
            return 0.0
        if isinstance(self.data, float):
            return self.data / self.steps
        elif isinstance(self.data, list):
            return [v/self.steps for v in self.data]
        else:
            return {k:(v/self.steps) for k, v in self.data.items()}


class Logger(object):
    """A wrapper class of get/set logger."""

    @staticmethod
    def get(name=None):
        """Get the logger by name.
        
        Args:
            name (str): the name of logger you want get. the default 
            is the root logger.

        Returns: the logger you want it.

        Note: Maybe you should invoke set_logger() before get it.
        """
        return logging.getLogger(name)

    @staticmethod
    def set(log_file, name=None, level=logging.INFO):
        """Set the logger to log info in console and file `log_file`.

        Args:
            log_file (path): the path of log file.
            name (str): logger name, `None` means the root logger.

        Returns: the logger.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # just add handlers for logger only once
        if not logger.hasHandlers():
            # Logging to a file
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fmt = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
            fh.setFormatter(fmt)
            logger.addHandler(fh)
            # Logging to console
            ch = logging.StreamHandler()
            ch.setLevel(logging.ERROR)
            ch.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(ch)
        return logger


class Serialization(object):
    """Save/load the model and optimizer parameters."""

    def __init__(self, checkpoint_dir=None):
        """
        Args:
            checkpoint_dir (path): the directory of checkpoint files.
        """
        self.checkpoint_dir = os.getcwd()  # current working directory
        if checkpoint_dir is not None:
            self.checkpoint_dir = checkpoint_dir
        self.name = '{}.pth.tar'  # PyTorch save/load format
        self.best = 'best.pth.tar'  # best model checkpoint filename
        self.logger = Logger.get()

    def serialize(self, model, epoch, optimizer=None, 
                  checkpoint='last', is_best=False):
        """Save model, optimizer and other parameters to file.

        Args:
            epoch (int): the epoch of training.
            checkpoint (str): the name of checkpoint, i.e., "last", "best".
            is_best (bool): whether model is the best so far.
        """
        if not os.path.isdir(self.checkpoint_dir):
            msg = "Checkpoint Directory does not exist! Making directory {}"
            self.logger.info(msg.format(self.checkpoint_dir))
            os.makedirs(self.checkpoint_dir)
        else:
            self.logger.info("Checkpoint Directory exists!")
        checkpoint_file = os.path.join(self.checkpoint_dir, 
                                       self.name.format(checkpoint))
        data = {
            'epoch': epoch,
            'model_state': model.state_dict()
        }
        if optimizer is not None:
            data['optim_state'] = optimizer.state_dict()
        # save checkpoint
        torch.save(data, checkpoint_file)
        msg = "Save model parameters into file: {}"
        self.logger.info(msg.format(checkpoint_file))
        # copy best model
        best_file = os.path.join(self.checkpoint_dir, self.best)
        if is_best and not os.path.samefile(checkpoint_file, best_file):
            shutil.copy(checkpoint_file, best_file)
            msg = "Save best model parameters into file: {}"
            self.logger.info(msg.format(best_file))

    def restore(self, model, optimizer=None, checkpoint='best'):
        """Restore model and optimizer from file.
        
        Args:
            checkpoint (str): the checkpoint name.

        Returns: `True` means success and `False` means failure.
        """
        if not os.path.isdir(self.checkpoint_dir):
            self.logger.error("Checkpoint Directory not exists! ")
            return False
        checkpoint_file = os.path.join(self.checkpoint_dir,
                                       self.name.format(checkpoint))
        if not os.path.isfile(checkpoint_file):
            self.logger.error("Checkpoint File not exists!")
            return False
        # restore checkpoint
        data = torch.load(checkpoint_file)
        model.load_state_dict(data['model_state'])
        if optimizer is not None:
            optimizer.load_state_dict(data['optim_state'])
        msg = "Restore model(epoch: {}) from file: {}"
        self.logger.info(msg.format(data['epoch'], checkpoint_file))
        return True


class ProgressBarWrapper(object):
    """A wrapper of iterator, shows the progress bar of iteration."""

    def __init__(self, iterator, iterator_len, with_bar=True, 
                 with_index=True, prefix='', suffix='', decimals=1, 
                 bar_len=50, fill='@', padding='-'):
        """
        Args:
            iterator (iterator): the data iterator.
            iterator_len (int): the length of the iterator.
            prefix (str): the prefix string for progress bar.
            suffix (str): the suffix string for progress bar.
            decimals (int): the number of decimals in percent.
            bar_len (int): the length of progress bar.
            fill (str): bar fill character.
            padding (str): bar padding character.
        """
        self.iterator = iterator
        self.iterator_len = iterator_len
        self.with_index = with_index
        self.with_bar = with_bar
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.bar_len = bar_len
        self.fill = fill
        self.padding = padding

    def set_prefix(self, prefix=''):
        self.prefix = prefix

    def set_suffix(self, suffix=''):
        self.suffix = suffix

    def print_bar(self, i):
        total = self.iterator_len
        percent = (i / float(total)) if total != 0 else 1.0
        fill_len = int(self.bar_len * percent)
        padding_len = self.bar_len - fill_len
        percent_fmt = ''.join([
            '{', '0:.{}f'.format(self.decimals), '}'
        ])
        percent_str = percent_fmt.format(100 * percent)
        bar_str = ''.join([self.fill]*fill_len + [self.padding]*padding_len)
        print()
        print('{} [{}] {}%% {}'.format(
            self.prefix, bar_str, percent_str, self.suffix))

    def __iter__(self):
        """Wrap the original iterator.

        Returns: a tuple of (index, bar, item)
        """
        for i, item in enumerate(self.iterator, 1):
            self.print_bar(i)
            if self.with_index and self.with_bar:
                yield i-1, self, item
            elif self.with_index and (not self.with_bar):
                yield i-1, item
            elif self.with_bar and (not self.with_index):
                yield self, item
            else:
                yield item
