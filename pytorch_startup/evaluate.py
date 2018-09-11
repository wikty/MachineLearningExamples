import os
import json
import argparse

import torch

import config
from model import model_factory
from data_loader import DataLoader
from utils import Params, Logger, RunningAvg, dump_to_json, Serialization


def evaluate(model, dateset, loss_fn, metrics):
    # set model to evaluation mode
    model.eval()

    logger = Logger.get()
    running_avg = RunningAvg()
    for batch in dateset:
        inputs, targets = batch
        stat = {}
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            stat['loss'] = loss.item()
            for name, metric in metrics.items():
                stat[name] = metric(outputs, targets).item()
        running_avg.step(stat)
    metrics_mean = running_avg()
    
    logger.info("- Evaluation metrics:")
    for name, value in metrics_mean.items():
        logger.info('    * {}: {:05.3f}'.format(name, value))

    return metrics_mean


if __name__ == '__main__':
    data_dir = config.data_dir
    model_dir = config.base_model_dir
    params_filename = config.params_file
    log_filename = config.evaluate_log
    datasets_params_file = config.datasets_params_file
    test_name = config.test_name
    best_checkpoint = 'best'
    last_checkpoint = 'last'
    checkpoint_filename = config.checkpoint_format
    best_metrics_filename = config.best_metrics_format
    last_metrics_filename = config.last_metrics_format   

    # define command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', 
                        default=data_dir, 
                        help="The directory containing the datasets")
    parser.add_argument('--model-dir', 
                        default=model_dir, 
                        help="The directory containing hyperparameters \
                        config file {}".format(params_filename))
    parser.add_argument('--checkpoint', 
                        default=best_checkpoint,
                        help="The name of checkpoint to restore model")
    parser.add_argument('--dataset-name',
                        default=test_name,
                        help="The name of dataset that want to evaluate")

    # parse command line arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    checkpoint = args.checkpoint
    dataset_name = args.dataset_name
    msg = "Data directory not exists: {}"
    assert os.path.isdir(data_dir), msg.format(data_dir)
    msg = "Model directory not exists: {}"
    assert os.path.isdir(model_dir), msg.format(model_dir)
    params_file = os.path.join(model_dir, params_filename)
    msg = "Model config file not exists: {}"
    assert os.path.isfile(params_file), msg.format(params_file)
    checkpoint_file = os.path.join(model_dir, 
                                   checkpoint_filename.format(checkpoint))
    msg = "Checkpoint file not exists: {}"
    assert os.path.isfile(checkpoint_file), msg.format(checkpoint_file)
    if checkpoint == best_checkpoint:
        metrics_filename = best_metrics_filename.format(dataset_name)
    else:
        metrics_filename = last_metrics_filename.format(dataset_name)

    # set logger
    logger = Logger.set(os.path.join(model_dir, log_filename))

    # load model configuration  
    logger.info("Loading the experiment configurations...")  
    params = Params(params_file)
    # cuda flag
    params.set('cuda', torch.cuda.is_available())
    logger.info("- done.")

    # load datesets
    logger.info("Loading the {} dataset...".format(dataset_name))
    datasets_params = Params(datasets_params_file)
    loader = DataLoader(data_dir, datasets_params, encoding='utf8')
    dataset = loader.load(dataset_name,
                          encoding='utf8',
                          batch_size=params.batch_size,
                          to_tensor=True,
                          to_cuda=params.cuda)
    logger.info("- done.")

    # add datasets parameters into params
    params.update(datasets_params)

    # create model, optimizer and so on.
    model, optimizer, criterion, metrics = model_factory(params)

    # restore model, optimizer
    status = Serialization(checkpoint_dir=model_dir).restore(
        model=model, checkpoint=checkpoint)
    
    if not status:
        logger.error("Restore model from the checkpoint: {}, failed".format(
            checkpoint))

    logger.info("Starting evaluate model on test dataset...")
    metrics_result = evaluate(model, dataset, criterion, metrics)
    logger.info("- done.")

    logger.info("Save metrics results...")
    metrics_file = os.path.join(model_dir, 
                                metrics_filename.format(checkpoint))
    dump_to_json(metrics_result, metrics_file)
    logger.info("- done.")
