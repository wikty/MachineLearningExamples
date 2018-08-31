import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt


class Network(nn.Module):
    """Assume the input image is 32x32 with 1-channel."""

    def __init__(self):
        super().__init__()
        # first conv layer: 
        # input 1-channel image, output 6-channel feature maps.
        # convolution kernel size is 5x5. gap is 1.
        self.conv1 = nn.Conv2d(1, 6, (5, 5))
        # second conv layer:
        # input 6-channel feature maps, output 16 feature maps.
        # convolution kernel size is 5x5. gap is 1.
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        # Full connection layers:
        # the last convolution layer output 16 feature maps of 5x5 size
        feature_num = 16*5*5
        self.fc1 = nn.Linear(feature_num, 120)  # affine transform
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """`x` is a mini-batch of samples, not a single sample."""
        # max pooling window size is 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size()[0], -1) # flatten feature maps
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def one_hot_to_int(t):
    return torch.argmax(t)


def int_to_one_hot(i, dim):
    t = torch.zeros(dim)
    t[i] = 1
    return t


class OneHotTransform(object):
    """transform the integer target into one-hot encoding target."""

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, sample):
        idx = sample.item()
        return int_to_one_hot(idx, self.dim)    



def load_data(path, batch_size=4, num_workers=2):
    # a sequential transforms for PIL Image
    transform = transforms.Compose([
        # Resize the input PIL Image to the given size.
        transforms.Resize((32, 32)),
        # Convert PIL Image (HxWxC) in range [0, 255] into
        # tensor (CxHxW) in range [0.0, 1.0].
        transforms.ToTensor(),
        # Normalization: change the range from [0.0, 1.0] to
        # [-1.0, 1.0]
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    # a sequential transforms for target/label.
    # target_transform = transforms.Compose([
    #     # change integer label into one-hot label
    #     OneHotTransform(10),
    # ])
    target_transform = None

    trainset = torchvision.datasets.MNIST(root=path, train=True,
                                          download=True, transform=transform,
                                          target_transform=target_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.MNIST(root=path, train=False,
                                         download=True, transform=transform,
                                         target_transform=target_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    classes = [str(i) for i in range(0, 10)]

    return trainloader, testloader, classes


def show_image(dataloader, classes):
    # get a batch of images and labels
    images, labels = iter(dataloader).next()
    # print(images.size(), labels.size())
    
    # Make a grid of images: 4D mini-batch Tensor of shape (B x C x H x W) or 
    # a list of images all of the same size.
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()  # tensor to NumPy ndarray
    npimg = np.transpose(npimg, (1, 2, 0))  # (CxHxW) to (HxWxC)
    plt.imshow(npimg)
    # print('Labels: %s' % ' '.join([
    #     classes[torch.argmax(label)] for label in labels]))
    print('Labels: %s' % ' '.join([
        classes[label] for label in labels]))
    plt.show()



if __name__ == '__main__':
    # hyperparameter
    epoch = 10
    lr = 6e-10
    batch_size = 100
    numbers = 10

    data_dir = '../data/MNIST'
    trainloader, testloader, classes = load_data(data_dir, batch_size)

    # show image
    # show_image(trainloader, classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))

    net = Network()
    print('Net: {}'.format(net))
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    # training
    for e in range(10):
        running_loss = 0.0
        for i, batch in enumerate(trainloader, 0):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i != 0) and (i % 1000 == 0):
                print('Epoch/Batch: {}/{}, Loss: {}'.format(
                    e, i, running_loss/1000.0
                ))
                running_loss = 0.0

    # evaluation
    total, correct = [0]*numbers, [0]*numbers
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            outputs = net(images)

            for i, j in zip(torch.argmax(outputs, dim=1), labels):
                i, j = i.item(), j.item()
                total[j] += 1
                if i == j:
                    correct[j] += 1

    print('Total Acc: %f%%' % (100.0*sum(correct)/sum(total)))

    for i in range(numbers):
        print('Number[%d](%d/%d) Acc: %f%%' % (
            i, correct[i], total[i], 100.0*correct[i]/total[i]
        ))
    

    # lr = 0.001

    # net = Network()
    # criterion = nn.MSELoss()
    # optimizer = optim.SGD(net.parameters(), lr=lr)

    # print(net)
    # print(net.parameters())

    # # a mini-batch with one sample
    # inputs = torch.randn(1, 32, 32).unsqueeze(0)
    # # inputs = torch.randn(1, 1, 32, 32)
    # targets = torch.randn(1, 10).unsqueeze(0)
    # outputs = net.forward(inputs)
    # print(outputs)
    # # net.zero_grad()
    # optimizer.zero_grad()
    # # outputs.backward(torch.randn(1, 10))
    # loss = criterion(outputs, targets.view(1, 10))
    # print(loss)
    # loss.backward()
    # print(net.conv1.bias.grad)
    # print(net.conv2.bias.grad)
    # # update parameters
    # # for param in net.parameters():
    # #     param.data.sub_(lr * param.grad.data)
    # optimizer.step()