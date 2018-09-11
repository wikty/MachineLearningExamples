import os
import json
import argparse

import config
from utils import load_from_json, Table


def extract(metrics, model_dir, params_filename, 
            metrics_filename):
    metrics_file = os.path.join(model_dir, metrics_filename)
    params_file = os.path.join(model_dir, params_filename)
    if os.path.isfile(metrics_file):
        data = load_from_json(metrics_file)
        data.update(load_from_json(params_file))
        data['path'] = model_dir
        metrics.append(data)

    for subitem in os.listdir(model_dir):
        subdir = os.path.join(model_dir, subitem)
        if not os.path.isdir(subdir):
            continue
        extract(metrics, subdir, params_filename, 
                metrics_filename)


if __name__ == '__main__':
    model_dir = config.base_model_dir
    params_filename = config.params_file
    metrics_filename = config.best_metrics_on_val_file

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', 
                        default=model_dir, 
                        help="The directory containing experiment results.")
    parser.add_argument('--find-best',
                        default=False,
                        help="Flag to enable to find the best mode.",
                        type=bool)
    parser.add_argument('--output-format',
                        choices=['table', 'csv'],
                        default='table',
                        help="The format of output.")

    args = parser.parse_args()
    model_dir = args.model_dir
    output_format = args.output_format
    find_best = args.find_best
    assert os.path.isdir(model_dir), "Model directory not exists!"

    metrics = Table()
    extract(metrics, model_dir, params_filename, 
            metrics_filename)

    if find_best:
        row = metrics.max('accuracy')
        print(row['path'])
        exit(0)

    # print summary information to console
    if output_format == 'table':
        print(metrics.tabulate())
    elif output_format == 'csv':
        print(metrics.csv())
    else:
        print('Invalid output format!')