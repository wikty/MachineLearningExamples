"""
A character-level RNN to classify words.

A character-level RNN reads words as a series of characters - outputting a
prediction and “hidden state” at each step, feeding its previous hidden state
into each next step. We take the final prediction to be the output, i.e. which
class the word belongs to.

Here we’ll train on a few thousand surnames from 18 languages of origin, and 
predict which language a name is from based on the spelling.
"""
import os
import sys
import glob
import string
import random
import unicodedata

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Vocab(object):

    def __init__(self, letters):
        self.index = {}
        for letter in set(letters):
            self.index[letter] = len(self.index)

    def __len__(self):
        return len(self.index)

    def __contains__(self, letter):
        return letter in self.index

    def size(self):
        return len(self)

    def encode(self, text):
        """Return one-hot encoding for text.
        
        Return: 
            a tensor with the shape (len(text), 1, vocab_size)"""
        tensor = torch.zeros(len(text), 1, self.size())
        for i, letter in enumerate(text):
            idx = self.index.get(letter, None)
            if idx is None:
                raise Exception('Invalid letter "{}"'.format(letter))
            tensor[i][0][idx] = 1
        return tensor


class Dataset(object):

    def __init__(self, data_dir, vocab):
        self.data_dir = data_dir
        self.vocab = vocab
        self.categories = {}
        self.surnames = {}
        # load data
        self.load_data()

    @property
    def num_categories(self):
        return len(self.categories)

    @property
    def all_categories(self):
        return list(self.categories.keys())

    def get_category(self, i):
        for category, j in self.categories.items():
            if i == j:
                return category
        return None

    def clean_line(self, line):
        """Turn a Unicode string to plain ASCII.
        http://stackoverflow.com/a/518232/2809427
        """
        return ''.join([
            c for c in unicodedata.normalize('NFD', line)
            if (unicodedata.category(c) != 'Mn'
                and c in self.vocab)
        ])

    def load_data(self):
        for filename in glob.glob(os.path.join(self.data_dir, '*.txt')):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.surnames[category] = []
            self.categories[category] = len(self.categories)
            with open(filename, 'r', encoding='utf8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    self.surnames[category].append(self.clean_line(line))

    def get_example(self):
        category = random.choice(list(self.categories.keys()))
        surname = random.choice(self.surnames[category])
        category_tensor = torch.tensor([self.categories[category]], 
                                       dtype=torch.long)
        # Note: each surname has different length
        surname_tensor = self.vocab.encode(surname)
        return surname, category, surname_tensor, category_tensor


class Net(nn.Module):
    """RNN model:
                 o1       o2     output   
                 |        |        |
            h0 -[ ]- h1 -[ ]- h2 -[ ]
                 |        |        |
                 i1       i2       i3

        input = [i1, i2, i3]
        output = one-hot for the category label
        h1 = linear1(cat(i1, h0))
        o1 = log_softmax(linear2(cat(i1, h0)))
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.combined_size = self.input_size + self.hidden_size
        # current input and hidden state to the next hidden state
        self.i2h = nn.Linear(self.combined_size, self.hidden_size)
        # current input and hidden state to the next output
        self.i2o = nn.Linear(self.combined_size, self.output_size)
        
    def init_hidden(self, device=None):
        return torch.zeros(1, self.hidden_size, device=device)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=-1)  # -1 dim is the last axis
        combined = combined.view(-1, combined.size(-1))
        hidden = self.i2h(combined)
        output = F.log_softmax(self.i2o(combined), dim=1)
        return output, hidden


def to_category(output, k=1):
    values, indices = output.topk(k=k)
    
    if k == 1:    
        return values[0].item(), indices[0].item()
    else:
        return [v.item() for v in values[0]], [i.item() for i in indices[0]]


def train(model, dataset, criterion, optimizer, model_path,
         device=None, num_iters=100000, print_step=100):
    running_loss = 0.0
    loss_history = []
    for i in range(num_iters):
        surname, category, stensor, ctensor = dataset.get_example()

        if device:
            stensor, ctensor = stensor.to(device), ctensor.to(device)

        hidden = model.init_hidden(device)
        for j in range(stensor.size(0)):
            # character-level input
            input = stensor[j]
            # RNN unfold to forward net
            output, hidden = model(input, hidden)

        loss = criterion(output, ctensor)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % print_step == 0:
            category = dataset.get_category(ctensor.item())
            guess = dataset.get_category(to_category(output)[1])
            avg_loss = running_loss / print_step
            loss_history.append(avg_loss)
            running_loss = 0.0
            print('{}-Iteration:'.format(i+1))
            print('Surname: {}, Category: {}, Guess: {}, Avg Loss: {}'.format(
                surname, category, guess, avg_loss))

    plt.figure()
    plt.title("Train Loss vs. Iteration")
    plt.plot(loss_history)
    plt.show()

    save_model(model, model_path)

    return loss_history


def create_confusion_matrix(model, dataset, device=None, 
    num_examples=10000):
    """confusion matrix, indicating for every actual language (rows) which 
    language the network guesses (columns)."""
    confusion = torch.zeros(dataset.num_categories, dataset.num_categories)
    for i in range(num_examples):
        surname, category, stensor, ctensor = dataset.get_example()

        if device:
            stensor, ctensor = stensor.to(device), ctensor.to(device)

        hidden = model.init_hidden(device)
        for j in range(stensor.size(0)):
            input = stensor[j]
            output, hidden = model(input, hidden)

        category_i = ctensor.item()
        guess_i = to_category(output)[1]
        confusion[category_i][guess_i] += 1

    # normalize confusion matrix in each row
    for i in range(confusion.size(0)):
        confusion[i] /= confusion[i].sum()

    return confusion


def plot_confusion_matrix(model, dataset, device=None, 
    num_examples=10000):
    confusion = create_confusion_matrix(model, dataset, device, num_examples)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + dataset.all_categories, rotation=90)
    ax.set_yticklabels([''] + dataset.all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def predict(model, vocab, device=None, k=5):
    print('Please your surname:')
    surname = input().strip()
    tensor = vocab.encode(surname).to(device)
    hidden = model.init_hidden(device)
    for i in range(tensor.size(0)):
        output, hidden = model(tensor[i], hidden)
    
    scores, categories = to_category(output, k)
    for score, category_i in zip(scores, categories):
        print('Category: {}, Score: {}'.format(
            dataset.get_category(category_i), score))


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def load_model(model_path, **kwargs):
    model = Net(**kwargs)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


if __name__ == '__main__':
    learning_rate = 0.001
    num_iters = 100000
    print_step = 100
    num_confusion_examples = 10000
    data_dir = '../data/surnames/names'
    model_path = '../models/surname/rnn_surname_classification.pth'

    vb = Vocab(string.ascii_letters + " .,;'")
    dataset = Dataset(data_dir, vb)
    
    input_size = vb.size()
    hidden_size = 128
    output_size = dataset.num_categories
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if sys.argv[-1] == 'train':
        model = Net(input_size, hidden_size, output_size).to(device)
        criterion = F.nll_loss  # negative-log-likelihook
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        train(model, dataset, criterion, optimizer, model_path,
            device=device, num_iters=num_iters, print_step=print_step)
    else:
        model = load_model(model_path,
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size).to(device)
        if sys.argv[-1] == 'predict':
            predict(model, vb, device)
        elif sys.argv[-1] == 'stat':
            plot_confusion_matrix(model, dataset, device, 
                num_confusion_examples)
        else:
            print('please specify the task!')