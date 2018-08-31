import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt


class Network(nn.Module):
    """Assume the input image is 32x32 with 3-channels."""

    def __init__(self):
        super().__init__()
        # in: 3 channels image, out: 6 feature maps
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        # in: 6 feature maps, out: 16 feature maps
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        # max-pooling size is 2x2
        self.pool = nn.MaxPool2d(2, 2)
        # the last conv layer output 16 feature maps of 5x5 size
        feature_num = 16*5*5
        self.fc1 = nn.Linear(feature_num, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """`x` is a mini-batch of samples, not a single sample."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten feature maps and keep the batch size/dim
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data(path, batch_size=4, num_workers=2):
    transform = transforms.Compose([
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range 
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the 
        # range [0.0, 1.0].
        transforms.ToTensor(),
        # Normalize a tensor image with mean and standard deviation.
        # input[channel] = (input[channel] - mean[channel]) / std[channel]
        # So we want change range from [0.0, 1.0] to [-1.0, 1.0].
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def show_image(dataloader, classes):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()  # get a batch of images
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    plt.show()



if __name__ == '__main__':
    # load data
    data_dir = '../data/CIFAR-10'
    trainloader, testloader, classes = load_data(data_dir)

    # show a batch of images
    # show_image(trainloader, classes)
    # exit(0)

    # hyperparameter
    epoch = 2
    learning_rate = 0.001
    momentum = 0.9

    # CUDA device if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))

    # Convolutional network
    net = Network()
    print(net)
    # Move network to GPU if it is available
    # Please network to device before set its parameters to optimizer
    net.to(device)
    # Loss: Cross-Entorpy Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer: SGD with momentum
    optimizer = optim.SGD(net.parameters(), 
                          lr=learning_rate, momentum=momentum)

    # Training
    for eth in range(epoch):
        running_loss = 0.0
        # for-each all of mini-batches in trainset
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # move train data on GPU if it is available
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)
            
            # loss
            loss = criterion(outputs, labels)
            
            # backward
            loss.backward()
            
            # update parameters
            optimizer.step()

            # accumulate training running loss
            running_loss += loss.item()
            
            if (i != 0) and (i % 2000 == 0):
                print('Epoch:{}, Mini-batch: {}, Loss: {}'.format(
                    eth, i, running_loss/2000
                ))
                running_loss = 0.0
    
    print('Training Done!')

    # Evaluation
    total, correct = [0]*10, [0]*10
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs.data, 1)
            c = (labels==predictions).squeeze()
            for label, s in zip(labels, c):
                total[label] += 1
                correct[label] += s.item()
    
    print('Accuracy of the 10000 test images: %2d %%' % (
        100.0*sum(correct)/sum(total)))
    
    for i in range(10):
        print('Accuracy of %10s: %2d %%' % (
            classes[i], 100 * correct[i] / total[i]))

