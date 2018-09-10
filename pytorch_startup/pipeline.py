import os
import sys
import subprocess

import config


PYTHON = sys.executable


class Pipeline(object):

    def __init__(self, context={}):
        self.workers = []
        self.context = {}
        self.context.update(context)

    def append(self, worker, **worker_kwargs):
        self.workers.append((worker, worker_kwargs))

    def run(self, kwargs={}):
        self.context.update(kwargs)
        for worker, worker_kwargs in self.workers:
            results = worker(self.context, **worker_kwargs)
            if results is not None:
                msg = "worker must be return a dict or None."
                assert isinstance(results, dict), msg
                self.context.update(results)


def build_dataset(cxt):
    data_file = os.path.join(config.data_dir, 
                             cxt['data_file'])
    subprocess.run([PYTHON, 'build_dataset.py',
                    '--data-dir', config.data_dir,
                    '--data-file', data_file,
                    '--train-factor', str(config.train_factor),
                    '--val-factor', str(config.val_factor),
                    '--test-factor', str(config.test_factor),
                    '--min-count-word', str(config.min_count_word),
                    '--min-count-tag', str(config.min_count_tag)])


if __name__ == '__main__':
    p = Pipeline({
        'data_file': 'ner_dataset.csv',
    })
    p.append(build_dataset)
    p.run()