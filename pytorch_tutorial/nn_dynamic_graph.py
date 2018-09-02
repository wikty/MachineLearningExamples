import random

import torch
from torch import nn


class NetDynamic(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        # the layer contains parameters, if you re-use the layer many times,
        # means the parameters are share with those layers.
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the hidden_layer Module that many times to compute hidden layer
        representations. While hidden_layer Module share weights with those layers.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        a = self.in_layer(x).clamp(min=0.)
        for i in range(random.randint(0, 3)):
            a = self.hidden_layer(a).clamp(min=0.)
        return self.out_layer(a)


dtype = torch.float
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
print('Device: {}'.format(device))

epoch = 5000
train_n = 100
learning_rate = 1e-8
in_dim, hidden_dim, out_dim = 1000, 500, 10
x = torch.randn(train_n, in_dim, dtype=dtype, device=device)
y = torch.randn(train_n, out_dim, dtype=dtype, device=device)

net = NetDynamic(in_dim, hidden_dim, out_dim).cuda(device)
print(net)
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

for e in range(epoch):
    output = net(x)
    loss = loss_fn(output, y)
    print('Loss: {}'.format(loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()