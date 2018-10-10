import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, input_size, output_size, conv_layers=(6, 16), 
        linear_layers=(120, 84), receptive_size=(5, 5), pooling_size=(2, 2)):
        """
        Params:
        `input_size` is a tuple with three elements (channels, width, height).
        `output_size` is the number of output layer units.
        `conv_layers` is a tuple to specify the number of feature maps in each
        conv layer's.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.conv_layers = conv_layers
        self.linear_layers = linear_layers
        self.receptive_size = receptive_size
        self.pooling_size = pooling_size
        # convolution layers
        self.conv = nn.ModuleList()
        in_channels = input_size[0]
        out_channels = -1
        width, height = input_size[1], input_size[2]
        for out_channels in conv_layers:
            if (width <= self.receptive_size[0] or 
                height <= self.receptive_size[1]):
                print("You have specified too many conv layer.")
                break
            conv = nn.Conv2d(in_channels, out_channels, self.receptive_size)
            self.conv.append(conv)
            in_channels = out_channels
            width = int((width - self.receptive_size[0] + 1) / 2)
            height = int((height - self.receptive_size[1] + 1) / 2)
        # full-connection layers
        self.fc = nn.ModuleList()
        in_size = out_channels * width * height
        out_size = -1
        for out_size in linear_layers:
            fc = nn.Linear(in_size, out_size)
            in_size = out_size
            self.fc.append(fc)
        # output layer
        self.ol = nn.Linear(out_size, self.output_size)

    def forward(self, x):
        """`x` is a mini-batch of samples, not a single sample."""
        # conv
        for i in range(len(self.conv)):
            x = F.max_pool2d(F.relu(self.conv[i](x)), self.pooling_size)
        # resize
        x = x.view(x.size()[0], -1)
        # full-connection
        for i in range(len(self.fc)):
            x = F.relu(self.fc[i](x))
        # output
        x = self.ol(x)
        return x

if __name__ == '__main__':

    net = Net(input_size=(1, 32, 32), output_size=10)

    print(net)
    # learnable parameters(a tensor) of net's first convolution layer
    print(list(net.parameters())[0])

    x = torch.randn(1, 32, 32)
    # make a single sample into a mini-batch of samples
    x = x.unsqueeze(0)

    # forward
    out = net(x)

    # MSE loss
    target = torch.randn(10)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(out, target)
    print(out)
    print(target)
    print(loss.item())

    # backward
    net.zero_grad()     # zeroes the gradient buffers of all parameters
    print(net.conv[0].bias.grad)
    loss.backward()
    print(net.conv[0].bias.grad)

    # optimization
    optimizer = optim.SGD(net.parameters(), lr=0.01)  # create optimizer
    
    # training
    for i in range(10):
        # training loop
        optimizer.zero_grad()  # zero gradient buffers
        out = net(x)  # forward information
        loss = criterion(out, target)  # loss function
        loss.backward()  # backward population gradient
        optimizer.step()  # update paramters