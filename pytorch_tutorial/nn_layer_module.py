from collections import OrderedDict

import torch
from torch import nn


class NetReLU(nn.Module):

    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
        self.linears = torch.nn.ModuleList()
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            self.linears.append(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        a = x
        for linear in self.linears[:-1]:
            a = linear(a).clamp(min=0.)
        return self.linears[-1](a)


def get_model_1(sizes, device):
    # Create model from custom define Module
    model = NetReLU(sizes).cuda(device)
    return model


def get_model_2(sizes, device):
    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.
    # 
    # layers = []
    # for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
    #     layers.append(torch.nn.Linear(in_dim, out_dim))
    #     layers.append(torch.nn.ReLU())
    # model = torch.nn.Sequential(*layers)
    
    layers = OrderedDict()
    i = 1
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
        layers['linear{}'.format(i)] = torch.nn.Linear(in_dim, out_dim)
        layers['relu{}'.format(i)] = torch.nn.ReLU()
        i = i + 1
    model = torch.nn.Sequential(layers)
    
    return model.cuda(device)


dtype = torch.float
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

epoch = 5000
train_n = 100
learning_rate = 1e-6
in_dim, hidden_dim, out_dim = 1000, 500, 10
x = torch.randn(train_n, in_dim, dtype=dtype, device=device)
y = torch.randn(train_n, out_dim, dtype=dtype, device=device)

# net = get_model_1((in_dim, hidden_dim, hidden_dim, hidden_dim, out_dim), device)
net = get_model_2((in_dim, hidden_dim, hidden_dim, hidden_dim, out_dim), device)
print('Net: {}'.format(net))
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# train
for e in range(epoch):
    output = net(x)
    loss = loss_fn(output, y)
    print('Epoch: {}, Loss: {}'.format(e, loss.item()))
    optimizer.zero_grad()  # clear gradient buffer
    loss.backward()
    optimizer.step()