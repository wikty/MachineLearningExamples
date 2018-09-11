import os
import argparse

import torch

import config
from model import model_factory
from data_loader import DataLoader
from evaluate import evaluate
from utils import (Params, Logger, RunningAvg, dump_to_json, 
    Serialization, ProgressBarWrapper)


def train(model, optimizer, criterion, trainset, epoch, num_batches, 
          running_avg_steps):
    logger = Logger().get()
    # set model to training mode
    model.train()

    loss_avg = RunningAvg()
    metrics_avg = RunningAvg()
    # wrap the dataset to show a progress bar of iteration
    bar = ProgressBarWrapper(trainset, 
                             num_batches,
                             with_bar=False,
                             with_index=True,
                             prefix='Epoch-{}'.format(epoch),
                             suffix='loss=None')
    for i, batch in bar:
        # training
        inputs, targets = batch
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update train loss progress
        loss_avg.step(loss.item())
        bar.set_suffix('loss={:05.3f}'.format(loss_avg()))
        # compute metrics in every `running_avg_steps` steps
        if i % running_avg_steps == 0:
            stat = {'loss': loss.item()}
            with torch.no_grad():
                for name, metric in metrics.items():
                    stat[name] = metric(outputs, targets).item()
            metrics_avg.step(stat)

    metrics_mean = metrics_avg()
    logger.info("- Training metrics:")
    for name, value in metrics_mean.items():
        logger.info('    * {}: {:05.3f}'.format(name, value))


def run(model, optimizer, criterion, metrics, trainloader, valloader, 
        num_epochs, running_avg_steps, model_dir, checkpoint,
        best_metrics_file, last_metrics_file):
    logger = Logger.get()
    # restore from a checkpoint if provide it
    if checkpoint is not None:
        Serialization(model_dir).restore(model, optimizer, checkpoint)
    
    max_acc = 0.0
    for epoch in range(num_epochs):
        logger.info('Train Epoch - {}/{}'.format(epoch, num_epochs))
        trainset = trainloader(shuffle=True)
        trainsize = trainset.dataset_size
        num_batches = int(trainsize / trainset.batch_size)
        train(model, optimizer, criterion, trainset, epoch, num_batches, 
              running_avg_steps)
        valset = valloader(shuffle=True)
        metrics_result = evaluate(model, valset, criterion, metrics)

        is_best = False
        if metrics_result['accuracy'] >= max_acc:
            is_best = True
            max_acc = metrics_result['accuracy']
            logger.info("- Found new best accuracy")
            # Save best val metrics
            dump_to_json(metrics_result, best_metrics_file)
        else:
            # Save latest val metrics
            dump_to_json(metrics_result, last_metrics_file)

        Serialization(model_dir).serialize(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            checkpoint='last',
            is_best=is_best
        )


if __name__ == '__main__':
    data_dir = config.data_dir
    model_dir = config.base_model_dir
    params_filename = config.params_file
    log_filename = config.train_log
    datasets_params_file = config.datasets_params_file
    train_name = config.train_name
    val_name = config.val_name
    best_metrics_filename = config.best_metrics_on_val_file
    last_metrics_filename = config.last_metrics_on_val_file

    # define command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', 
                        default=data_dir, 
                        help="The directory contains datasets.")
    parser.add_argument('--model-dir', 
                        default=model_dir, 
                        help="The directory contains hyperparameters \
                        config file and will store log and result files.")
    parser.add_argument('--checkpoint', 
                        default=None,
                        help="The name of checkpoint to restore model.")

    # parse command line arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    checkpoint = args.checkpoint

    # check settings
    msg = "Data directory not exists: {}"
    assert os.path.isdir(data_dir), msg.format(data_dir)
    msg = "Model directory not exists: {}"
    assert os.path.isdir(model_dir), msg.format(model_dir)
    params_file = os.path.join(model_dir, params_filename)
    msg = "Model config file not exists: {}"
    assert os.path.isfile(params_file), msg.format(params_file)
    best_metrics_file = os.path.join(model_dir, best_metrics_filename)
    last_metrics_file = os.path.join(model_dir, last_metrics_filename)

    # set logger
    # Note: log file will be stored in the `model_dir` directory
    logger = Logger.set(os.path.join(model_dir, log_filename))

    # load model configuration  
    logger.info("Loading the experiment configurations...")  
    params = Params(params_file)
    # cuda flag
    params.set('cuda', torch.cuda.is_available())
    logger.info("- done.")

    # load datesets
    logger.info("Loading the datasets...")
    datasets_params = Params(datasets_params_file)
    loader = DataLoader(data_dir, datasets_params, encoding='utf8')
    # add datasets parameters into params
    params.update(datasets_params)
    # make dateset loaders
    def trainloader(shuffle=True):
        return loader.load(train_name,
                           encoding='utf8',
                           batch_size=params.batch_size,
                           to_tensor=True,
                           to_cuda=params.cuda,
                           shuffle=shuffle)
    def valloader(shuffle=True):
        return loader.load(val_name,
                           encoding='utf8',
                           batch_size=params.batch_size,
                           to_tensor=True,
                           to_cuda=params.cuda,
                           shuffle=shuffle)
    logger.info("- done.")
    # create model, optimizer and so on.
    model, optimizer, criterion, metrics = model_factory(params)
    # run train and evaluate
    num_epochs = params.num_epochs
    running_avg_steps = params.running_avg_steps
    logger.info("Starting training for {} epoch(s)".format(num_epochs))
    run(model, optimizer, criterion, metrics, trainloader, valloader, 
        num_epochs, running_avg_steps, model_dir, checkpoint,
        best_metrics_file, last_metrics_file)
    logger.info("- done.")

