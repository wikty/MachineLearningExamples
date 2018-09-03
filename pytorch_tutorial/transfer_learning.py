import os
import sys
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt



def load_data(image_dir, batch_size=4, num_workers=2):
    data_transforms = {
        'train': transforms.Compose([
            # Data augmentation
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # PIL image to tensor
            transforms.ToTensor(),
            # Normalization the 3-channels image
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    dataloaders, datasizes = {}, {}
    classes = None
    for name in ['train', 'val']:
        dirname = os.path.join(image_dir, name)
        image_datasets = datasets.ImageFolder(dirname, data_transforms[name])
        datasizes[name] = len(image_datasets)
        classes = image_datasets.classes
        # maybe num_workers should be set to 0 on Windows
        dataloaders[name] = torch.utils.data.DataLoader(image_datasets, 
            batch_size=batch_size, shuffle=True, num_workers=num_workers) 

    return dataloaders, datasizes, classes


def load_model(device, fixed_features=False, lr=0.01, momentum=0.9):
    # load pretrained model from torchvision
    model = models.resnet18(pretrained=True)
    # freeze parameters in the hidden layers, i.e., fix feature extractor
    if fixed_features:
        for param in model.parameters():
            param.requires_grad_(False)
    # reset the last fully connection layer
    in_dim = model.fc.in_features
    model.fc = nn.Linear(in_dim, 2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if fixed_features:
        # Observe that only parameters of final layer are being optimized
        optimizer = optim.SGD(model.fc.parameters(), 
            lr=lr, momentum=momentum)
    else:
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(model.parameters(), 
            lr=lr, momentum=momentum)

    # A learning rate scheduler
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return model, criterion, optimizer, exp_lr_scheduler


def train_model(model, criterion, optimizer, scheduler, dataloaders, 
    datasizes, device, epochs=50):
    best_acc = 0.0
    best_model_state  = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for e in range(epochs):
        print('Epoch: {}/{}'.format(e, epochs-1))
        print('-'*10)

        # training and validation in every training epoch
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                # change learning rate
                scheduler.step()
                # set model to train mode
                model.train()
            else:
                # set model to evaluate mode
                model.eval()

            # for-each mini-batch
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # forward
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # backward
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / datasizes[phase]
            epoch_acc = running_corrects.double() / datasizes[phase]

            print('{}| Loss: {:.4f}, Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc
            ))

            if phase == 'val' and epoch_acc > best_acc:
                best_model_state = copy.deepcopy(model.state_dict())
                best_acc = epoch_acc
            print()

    print('Train Complete in {:0.5f}s'.format(
        time.time() - start_time))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model parameters
    model.load_state_dict(best_model_state)
    return model


def visualize_model(model, dataloaders, class_names, device, 
    num_images=6):
    was_training = model.training
    model.eval()  # set model to evaluate mode
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)  # recover model mode
                    return
        # recover model mode
        model.train(mode=was_training)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    # convert (CxHxW) to (HxWxC)
    inp = inp.numpy().transpose((1, 2, 0))
    # unnormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    # clip value into [0, 1] range
    inp = np.clip(inp, 0, 1)
    # show PIL image
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # plt.pause(1)  # pause a bit so that plots are updated


def show_images(inputs, classes, class_names):
    # Make a grid from the batch of images
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])



if __name__ == '__main__':

    data_dir = '../data/hymenoptera_data'
    epochs = 50
    momentum = 0.9
    learning_rate = 0.01

    # plt.ion()  # interactive mode enable

    device = None
    if sys.argv[1:] and sys.argv[1] == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print('Device: {}'.format(device))

    # load data
    dataloaders, datasizes, class_names = load_data(data_dir)

    # show a batch of training images
    # show_images(*next(iter(dataloaders['train'])), class_names)

    # load a pre-trained model and reset it for finetuning
    model, criterion, optimizer, scheduler = load_model(
        device, fixed_features=True, lr=learning_rate, momentum=momentum
    )

    # training/finetuning model
    model = train_model(model, criterion, optimizer, scheduler, dataloaders,
        datasizes, device, epochs=epochs
    )

    # visualize model
    visualize_model(model, dataloaders, class_names, device, 
        num_images=8
    )

    # plt.ioff()  # interactive mode disable
    plt.show()

