import os
import sys
import shutil
import argparse
import subprocess


import config
from utils import Params


def train_model(model_dir, data_dir, checkpoint):
    args = [sys.executable, 
            'train.py',
            '--model-dir', model_dir,
            '--data-dir', data_dir]
    if checkpoint is not None:
        args.extend(['--checkpoint', checkpoint])
    subprocess.run(args, check=True, shell=True)


def search_all(hyperparams, parent_dir, data_dir, checkpoint,
               params_filename):
    assert isinstance(hyperparams, dict)
    params_file = os.path.join(parent_dir, params_filename)
    msg = "Parent directory doesn't contain params file."
    assert os.path.isfile(params_file), msg

    # create experiment directory and copy params file
    hyperparams_dir = os.path.join(parent_dir, 'hyperparams')
    if os.path.isdir(hyperparams_dir):
        shutil.rmtree(hyperparams_dir)
    os.makedirs(hyperparams_dir)
    shutil.copy(params_file, 
                os.path.join(hyperparams_dir, params_filename))

    pds = [hyperparams_dir]
    for name, candidates in hyperparams.items():
        new_pds = []
        for pd in pds:
            model_dirs = search_hyperparam((name, candidates), 
                pd, data_dir, checkpoint, params_filename)
            new_pds.extend(model_dirs)
        pds = new_pds
    return hyperparams_dir


def search_hyperparam(hyperparam, parent_dir, data_dir, checkpoint,
                      params_filename):
    assert isinstance(hyperparam, tuple)
    params_file = os.path.join(parent_dir, params_filename)
    msg = "Parent directory doesn't contain params file."
    assert os.path.isfile(params_file), msg

    params = Params(params_file)
    name, candidates = hyperparam
    model_dirs = []
    for candidate in candidates:
        experiment = '{}_{}'.format(name, candidate)
        model_dir = os.path.join(parent_dir, experiment)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        model_dirs.append(model_dir)
        # create params file for the experiment
        params.set(name, candidate)
        params.dump(os.path.join(model_dir, params_filename))
        # run subprocess to train model
        train_model(model_dir, data_dir, checkpoint)
    return model_dirs


if __name__ == '__main__':
    data_dir = config.data_dir
    model_dir = config.base_model_dir
    params_filename = config.params_file

    # define command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', 
                        default=data_dir, 
                        help="The directory containing the datasets")
    parser.add_argument('--model-dir', 
                        default=model_dir, 
                        help="The directory contains hyperparameters \
                        config file and will store log and result files.")
    parser.add_argument('--checkpoint', 
                        default=None,
                        help="The name of checkpoint to restore model")
    parser.add_argument('--job',
                        choices=['lr', 'bz', 'ed', 'all'],
                        help="The hyperparameters name want to search.")

    # parse command line arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    checkpoint = args.checkpoint
    job = args.job
    msg = "Data directory not exists: {}"
    assert os.path.isdir(data_dir), msg.format(data_dir)
    msg = "Model directory not exists: {}"
    assert os.path.isdir(model_dir), msg.format(model_dir)
    params_file = os.path.join(model_dir, params_filename)
    msg = "Model config file not exists: {}"
    assert os.path.isfile(params_file), msg.format(params_file)

    # config your experiment learning rates
    hyperparams = {
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'batch_size': [10, 50, 100],
        'embedding_dim': [50, 100, 200]
    }
    if job == 'lr':
        search_hyperparam(('learning_rate', hyperparams['learning_rate']),
                          model_dir, data_dir, checkpoint, params_filename)
    elif job == 'bz':
        search_hyperparam(('batch_size', hyperparams['batch_size']),
                          model_dir, data_dir, checkpoint, params_filename)
    elif job == 'ed':
        search_hyperparam(('embedding_dim', hyperparams['embedding_dim']),
                          model_dir, data_dir, checkpoint, params_filename)
    elif job == 'all':
        search_all(hyperparams, model_dir, data_dir, checkpoint, 
                   params_filename)
    else:
        print('Please specify the valid job name!')
    