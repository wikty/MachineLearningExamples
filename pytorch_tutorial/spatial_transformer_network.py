"""
Spatial transformer networks are a generalization of differentiable attention
to any spatial transformation. Spatial transformer networks (STN for short) 
allow a neural network to learn how to perform spatial transformations on the
input image in order to enhance the geometric invariance of the model. For 
example, it can crop a region of interest, scale and correct the orientation 
of an image. It can be a useful mechanism because CNNs are not invariant to 
rotation and scale and more general affine transformations.

One of the best things about STN is the ability to simply plug it into any 
existing CNN with very little modification.

More see the paper: https://arxiv.org/abs/1506.02025
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    """
    A standard convolutional network augmented with a spatial transformer 
    network for MNIST classification task.

    Spatial transformer networks boils down to three main components :
    
    * The localization network is a regular CNN which regresses the transformation parameters. The transformation is never learned explicitly from this dataset, instead the network learns automatically the spatial transformations that enhances the global accuracy.
    * The grid generator generates a grid of coordinates in the input image corresponding to each pixel from the output image.
    * The sampler uses the parameters of the transformation and applies it to the input image.
    """

    def __init__(self):
        super().__init__()
        # The standard convolutional network
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # Spatial transform the input
        x = self.stn(x)

        # Perform the usual CNN forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_data(image_dir, batch_size=64, shuffle=True, 
              num_workers=2):
    dataset_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    datasetloaders = {}
    for name, is_train in zip(('train', 'val'), (True, False)):
        datasetloaders[name] = torch.utils.data.DataLoader(
        datasets.MNIST(root=image_dir, train=is_train, download=True, 
                       transform=dataset_tfm), 
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return datasetloaders


def train_model(model, dataloaders, criterion, optimizer, num_epochs, 
                device=None):
    start_time = time.time()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch: {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to train mode
            else:
                model.eval()  # set model to eval mode

            running_loss = 0.0
            running_corrects = 0

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

                _, preds = torch.max(outputs, dim=1)
                running_loss += loss.item() * inputs.size(0)
                # preds = outputs.max(1, keepdim=True)[1]
                # running_corrects += preds.eq(labels.view_as(preds)).sum().item()
                running_corrects += torch.sum(preds==labels)

            dataset_len = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_len
            epoch_acc = running_corrects.double() / dataset_len
            print('{} Loss: {:.4f}, Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

    print()
    elapsed_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        elapsed_time // 60, elapsed_time % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn(model, dataloaders, device=None):
    # We want to visualize the output of the spatial transformers layer
    # after the training, we visualize a batch of input images and
    # the corresponding transformed batch using STN.
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(dataloaders['val']))[0]
        if device:
            data = data.to(device)

        input_tensor = data.cpu()
        output_tensor = model.stn(data).cpu()  # STN transform

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(output_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')



if __name__ == '__main__':
    plt.ion()   # interactive mode

    data_dir = '../data/MNIST'
    dataloaders = load_data(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)
    criterion = F.nll_loss
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_model(model, dataloaders, criterion, optimizer, 10, device)

    visualize_stn(model, dataloaders, device)

    plt.ioff()
    plt.show()