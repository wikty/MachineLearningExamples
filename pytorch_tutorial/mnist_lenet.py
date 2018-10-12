import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


class Net(nn.Module):
    """LeNet model."""

    def __init__(self):
        """Assume the input image is 28x28 with 1-channel."""
        super().__init__()
        # Conv-1
        # In: 1@28x28, Out: 10@24x24
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        
        # MaxPool-1
        # In: 10@24x24, Out: 10@12x12

        # Conv-2
        # In: 10@12x12, Out: 20@8x8
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # MaxPool-2
        # In: 20@8x8, Out: 20@4x4

        # Dropout
        self.conv2_drop = nn.Dropout2d()

        # FullConnect-1
        # In: 20*4*4, Out: 50
        self.fc1 = nn.Linear(320, 50)

        # FullConnect-2
        # In: 50, Out: 10
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)  # 20*4*4
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_data(data_dir, batch_size=64, shuffle=True, num_workers=2):
    # a sequential transforms for PIL Image
    transform = transforms.Compose([
        # Convert PIL Image (HxWxC) in range [0, 255] into
        # tensor (CxHxW) in range [0.0, 1.0].
        transforms.ToTensor(),
        # Normalization: change the range from [0.0, 1.0] to
        # [-1.0, 1.0]
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # a sequential transforms for target/label.
    # target_transform = transforms.Compose([
    #     # change integer label into one-hot label
    #     OneHotTransform(10),
    # ])
    target_transform = None

    trainset = torchvision.datasets.MNIST(root=data_dir, train=True,
                                          download=True, transform=transform,
                                          target_transform=target_transform)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=batch_size,
                                              shuffle=shuffle, 
                                              num_workers=num_workers)

    testset = torchvision.datasets.MNIST(root=data_dir, train=False,
                                         download=True, transform=transform,
                                         target_transform=target_transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=batch_size,
                                             shuffle=shuffle, 
                                             num_workers=num_workers)

    classes = [str(i) for i in range(0, 10)]

    return classes, {'train': trainloader, 'val': testloader}


def train_model(model, dataloaders, criterion, optimizer, 
        save_path=None, device=None, num_epochs=10):
    """Train, validate and save model."""
    start_time = time.time()
    best_acc = 0.0
    best_model_state = model.state_dict()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            for inputs, labels in dataloaders[phase]:
                if device:
                    inputs, labels = inputs.to(device), labels.to(device)                

                with torch.set_grad_enabled(phase=='train'):
                    # forward
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()*inputs.size(0)
                    _, preds = torch.max(outputs, dim=1)
                    running_correct += torch.sum(preds==labels).item()

            size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / size
            epoch_acc = float(running_correct) / size
            print('{} Loss: {:.4f}, Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_state = model.state_dict()
    
    during = time.time() - start_time
    print()
    print('='*10)
    print('Train complete in {}m:{}s'.format(
        during // 60, during % 60))
    print('The best validation accuracy: {}'.format(best_acc))

    if save_path:
        torch.save(best_model_state, save_path)
        print('Save the model into {}'.format(save_path))


def load_model(save_path, training=False):
    model = Net()
    model.load_state_dict(torch.load(save_path))
    if training:
        model.train()
    else:
        model.eval()
    return model


if __name__ == '__main__':
    data_dir = '../data/MNIST'
    model_dir = '../models/MNIST'
    save_path = os.path.join(model_dir, 'lenet_mnist_model.pth')
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.01
    momentum = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Net().to(device)
    classes, dataloaders = load_data(data_dir, batch_size)
    criterion = F.nll_loss  # negative-log-likelihook
    optimizer = optim.SGD(model.parameters(), 
        lr=learning_rate, momentum=momentum)

    train_model(model, dataloaders, criterion, optimizer,
        save_path=save_path, device=device, num_epochs=num_epochs)