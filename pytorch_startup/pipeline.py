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
    subprocess.run([PYTHON, cxt['build_dataset_script'],
                    '--data-dir', config.data_dir,
                    '--data-file', data_file,
                    '--data-factor', str(cxt['data_factor']),
                    '--train-factor', str(config.train_factor),
                    '--val-factor', str(config.val_factor),
                    '--test-factor', str(config.test_factor),
                    '--min-count-word', str(config.min_count_word),
                    '--min-count-tag', str(config.min_count_tag)])


def train(cxt):
    model_dir = config.base_model_dir
    subprocess.run([PYTHON, cxt['train_script'],
                    '--data-dir', config.data_dir,
                    '--model-dir', model_dir])
    return {'model_dir': model_dir}


def refine(cxt):
    model_dir = cxt['model_dir']
    subprocess.run([PYTHON, cxt['refine_script'],
                   '--data-dir', config.data_dir,
                   '--model-dir', model_dir,
                   '--checkpoint', 'best',
                   '--job', 'lr'])
    return {'model_dir': model_dir}


def summary(cxt):
    rtn = subprocess.run([PYTHON, cxt['summary_script'],
                          '--model-dir', cxt['model_dir'],
                          '--find-best', str(True)], 
                          capture_output=True, text=True)
    best_mode_dir = rtn.stdout.strip()
    return {'model_dir': best_mode_dir}


def evaluate(cxt):
    subprocess.run([PYTHON, cxt['evaluate_script'],
                    '--data-dir', config.data_dir,
                    '--model-dir', cxt['model_dir']])
    
    print('The directory of the best model: {}'.format(
        cxt['model_dir']))


if __name__ == '__main__':
    p = Pipeline({
        'data_file': 'ner_dataset.csv',
        'data_factor': 0.05,
        'build_dataset_script': 'build_dataset.py',
        'train_script': 'train.py',
        'refine_script': 'hyperparams.py',
        'summary_script': 'summary.py',
        'evaluate_script': 'evaluate.py'
    })
    p.append(build_dataset)
    p.append(train)
    p.append(refine)
    p.append(summary)
    p.append(evaluate)
    p.run()