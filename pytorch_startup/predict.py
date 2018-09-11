import os
import argparse

import torch

import config
from data_loader import Vocabulary
from model import model_factory
from utils import Serialization, Params


def lines_align(lines, padding='-'):
    """
    Args:
        lines (list): a list of list of string.

    For Example:
    ```
    align_print([['hello', 'world'], ['O', 'B-org']])
    ```
    """
    data = {}
    outputs = []
    for line in lines:
        for i, item in enumerate(line):
            if i not in data:
                data[i] = []
            data[i].append(item)
        outputs.append([])
    for i in sorted(data.keys()):
        max_width = len(max(data[i], key=lambda s: len(s)))
        fmt = '{' + ':{}^{}'.format(padding, max_width) + '}'
        for row, item in enumerate(data[i]):
            outputs[row].append(fmt.format(item))
    return outputs


def encode(tokens, word_vocab, unk_word, to_cuda=None):
    """
    Args:
        tokens (list): a list of token string.
    """
    # to vocab index
    tokens_index = []
    for token in tokens:
        i = word_vocab.encode(token, default=None)
        if i is None:
            i = word_vocab.encode(unk_word)
        tokens_index.append(i)
    # to tensor
    inputs = torch.tensor(tokens_index, dtype=torch.long)
    if to_cuda:
        inputs = inputs.cuda()
    # to a mini-batch
    return inputs.unsqueeze(0)


def decode(predictions, tag_vocab):
    """
    Args:
        tags (Tensor): the shape is (tokens_len, tag_vocab_size).
    """
    predictions = torch.argmax(predictions, dim=1)
    tags = [tag_vocab.decode(p.item()) for p in predictions]
    return tags


def predict(model, word_vocab, tag_vocab, unk_word, 
            input_file, output_file, encoding, to_cuda):
    lines = []
    with open(input_file, 'r', encoding=encoding) as f:
        lines = f.readlines()
    with open(output_file, 'w', encoding=encoding) as f:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            inputs = encode(tokens, word_vocab, unk_word, to_cuda)
            predictions = model(inputs)
            tags = decode(predictions, tag_vocab)
            result = lines_align([tokens, tags])
            f.write(' '.join(result[0]) + '\n')
            f.write(' '.join(result[1]) + '\n')


if __name__ == '__main__':
    model_dir = config.base_model_dir
    data_dir = config.data_dir
    datasets_params_file = config.datasets_params_file
    params_filename = config.params_file
    words_txt = 'words.txt'
    tags_txt = 'tags.txt'

    # define command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', 
                        default=model_dir, 
                        help="The directory contains hyperparameters \
                        config file and will store log and result files.")
    parser.add_argument('--data-dir', 
                        default=data_dir, 
                        help="The directory contains datasets.")
    parser.add_argument('--checkpoint', 
                        default='best',
                        help="The name of checkpoint to restore model.")
    parser.add_argument('--input-file',
                        default='inputs.txt',
                        help="The input file for prediction.")
    parser.add_argument('--output-file',
                        default='outputs.txt',
                        help="The output file to save predictions.")
    parser.add_argument('--encoding',
                        default='utf8',
                        help="The encoding for input and output file.")

    # parse command line arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    checkpoint = args.checkpoint
    input_file = args.input_file
    output_file = args.output_file
    encoding = args.encoding

    msg = "Data directory not exists: {}"
    assert os.path.isdir(data_dir), msg.format(data_dir)
    msg = "Model directory not exists: {}"
    assert os.path.isdir(model_dir), msg.format(model_dir)
    msg = "Input file not exists: {}"
    assert os.path.isfile(input_file), msg.format(input_file)

    datasets_params = Params(datasets_params_file)
    word_vocab = Vocabulary(os.path.join(data_dir, words_txt))
    tag_vocab = Vocabulary(os.path.join(data_dir, tags_txt))
    unk_word = datasets_params.unk_word

    params = Params(os.path.join(model_dir, params_filename))
    params.update(datasets_params)
    params.set('cuda', torch.cuda.is_available())

    # restore model from the checkpoint
    model, *others = model_factory(params)
    Serialization(model_dir).restore(model, checkpoint=checkpoint)

    # predict
    predict(model, word_vocab, tag_vocab, unk_word, 
            input_file, output_file, encoding, params.cuda)

    print("It's done! Please check the output file:")
    print(output_file)