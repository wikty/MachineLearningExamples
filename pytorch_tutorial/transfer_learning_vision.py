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


def disable_autograd(model):
    for param in model.parameters():
        param.required_grad = False


def load_data(image_dir, input_size, batch_size=4, shuffle=True, 
              num_workers=2):
    data_transforms = {
        'train': transforms.Compose([
            # Data augmentation
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            # PIL image to tensor
            transforms.ToTensor(),
            # Normalization the 3-channels image
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    dataloaders, datasizes = {}, {}
    classes = None
    for name in ['train', 'val']:
        dirname = os.path.join(image_dir, name)
        image_datasets = datasets.ImageFolder(dirname, data_transforms[name])
        classes = image_datasets.classes
        datasizes[name] = len(image_datasets)
        # maybe num_workers should be set to 0 on Windows
        dataloaders[name] = torch.utils.data.DataLoader(image_datasets, 
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloaders, datasizes, classes


def load_model(model_name, num_classes, feature_extract, 
               device=None, pretrained=True):
    """Load and reshape model.

    Note: All of models in the torchvision are trained with ImageNet
    dataset, so they all have output layers of size 1000. Our dataset
    has two classes, so we should reshape the torchvision models to 
    suit for our case. You should know this is not an automatic procedure
    and is unique to each model.
    """
    model = None
    input_size = 0  # the input_size of model
    params = []  # parameters should be optimized

    if model_name == 'resnet':
        # Resnet was introduced in the paper "Deep Residual Learning for Image
        # Recognition". There are several variants of different sizes, 
        # including Resnet18, Resnet34, Resnet50, Resnet101, and Resnet152, all
        # of which are available from torchvision models.
        # 
        # Here we use Resnet18, the last layer of it is a a fully connected 
        # layer:
        # (fc): Linear(in_features=512, out_features=1000, bias=True)
        
        # Resnet18 requires the input size to be (224,224)
        input_size = 224
        # load model
        model = models.resnet18(pretrained=pretrained)
        if feature_extract:
            disable_autograd(model)
        # reshape model
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == 'alexnet':
        # Alexnet was introduced in the paper ImageNet Classification with 
        # "Deep Convolutional Neural Networks" and was the first very 
        # successful CNN on the ImageNet dataset.
        # The output layer:
        # (6): Linear(in_features=4096, out_features=1000, bias=True)
        
        # Alexnet18 requires the input size to be (224,224)
        input_size = 224
        # load model
        model = models.alexnet(pretrained=pretrained)
        if feature_extract:
            disable_autograd(model)
        # reshape model
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif model_name == 'vgg':
        # VGG was introduced in the paper "Very Deep Convolutional Networks
        # for Large-Scale Image Recognition". Torchvision offers eight 
        # versions of VGG with various lengths and some that have batch 
        # normalizations layers. 
        # 
        # Here we use VGG-11 with batch normalization. The output layer:
        # (6): Linear(in_features=4096, out_features=1000, bias=True)
        
        # VGG11_bn requires the input size to be (224,224)
        input_size = 224
        # load model
        model =models.vgg11_bn(pretrained=pretrained)
        if feature_extract:
            disable_autograd(model)
        # reshape model
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif model_name == "squeezenet":
        # The Squeeznet architecture is described in the paper "SqueezeNet: 
        # AlexNet-level accuracy with 50x fewer parameters and <0.5MB model 
        # size" and uses a different output structure than any of the other
        # models shown here.
        # Torchvision has two versions of Squeezenet, we use version 1.0.

        # Squeezenet requires the input size to be (224,224)
        input_size = 224
        # load model
        model = models.squeezenet1_0(pretrained=pretrained)
        if feature_extract:
            disable_autograd(model)
        # reshape model
        model.num_classes = num_classes
        model.classifier[1] = nn.Conv2d(512, num_classes, 
            kernel_size=(1,1), stride=(1,1))
    elif model_name == 'densenet':
        # Densenet was introduced in the paper "Densely Connected 
        # Convolutional Networks". Torchvision has four variants of Densenet.
        #
        # Here we only use Densenet-121. The output layer:
        # (classifier): Linear(in_features=1024, out_features=1000, bias=True)
        
        # Densenet requires the input size to be (224,224)
        input_size = 224
        # load model
        model = models.densenet121(pretrained=pretrained)
        if feature_extract:
            disable_autograd(model)
        # reshape model
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    elif model_name == 'inception':
        # Inception v3 was first described in "Rethinking the Inception 
        # Architecture for Computer Vision". This network is unique because it
        # has two output layers when training. The second output is known as 
        # an auxiliary output and is contained in the AuxLogits part of the 
        # network. The primary output is a linear layer at the end of the 
        # network. Note, when testing we only consider the primary output.
        
        # Inception v3 requires the input size to be (299, 299)
        input_size = 299
        # load model
        model = models.inception_v3(pretrained=pretrained)
        if feature_extract:
            disable_autograd(model)
        # reshape the auxilary net
        num_features = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_features, num_classes)
        # reshape the primary net
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        print("Invalid model name, exiting...")
        exit()

    # transfer to device
    if device:
        model = model.to(device)

    # collect the learnable parameters
    if not feature_extract:
        # finetune all of parameters
        params = model.parameters()
    else:
        # learn part of parameters
        params = [p for p in model.parameters() if p.requires_grad]

    return model, params, input_size


def train_model(model, dataloaders, criterion, optimizer, num_epochs, 
                device=None, is_inception_model=False):
    start_time = time.time()

    acc_history = []
    best_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # model set to train mode
            else:
                model.eval()  # model set to eval mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                if device:
                    inputs, labels = inputs.to(device), labels.to(device)

                with torch.set_grad_enabled(phase=='train'):
                    # forward
                    # track history if only in train
                    if is_inception_model and phase == 'train':
                        # Inception v3 model is a special case
                        # https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # backward 
                    # calculate gradients if only in train
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # statistics
                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item()*inputs.size(0)
                    running_corrects += torch.sum(preds==labels.data)

            dataset_len = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_len
            epoch_acc = running_corrects.double() / dataset_len

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase != 'train':
                acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_state = copy.deepcopy(model.state_dict())
    
    print()
    elapsed_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        elapsed_time // 60, elapsed_time % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_state)
    return model, acc_history


if __name__ == '__main__':
    # model_names = ['resnet', 'alexnet', 'vgg', 'squeezenet', 
    #                'densenet', 'inception']
    model_name = 'resnet'
    data_dir = '../data/hymenoptera_data'
    num_classes = 2
    batch_size = 4
    num_epochs = 50
    momentum = 0.9
    learning_rate = 0.001

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # train pretrained model
    model, params, input_size = load_model(model_name, num_classes, 
                                           False, device, True)
    dataloaders, datasizes, class_names = load_data(data_dir, input_size,
                                                    batch_size, True, 2)

    optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    model, hist = train_model(model, dataloaders, criterion, optimizer, 
                              num_epochs, device, (model_name=='inception'))
    hist1 = [h.cpu().numpy() for h in hist]

    # train non-pretrained model
    model, params, input_size = load_model(model_name, num_classes, 
                                           False, device, False)
    dataloaders, datasizes, class_names = load_data(data_dir, input_size,
                                                    batch_size, True, 2)
    optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    model, hist = train_model(model, dataloaders, criterion, optimizer, 
                              num_epochs, device, (model_name=='inception'))
    hist2 = [h.cpu().numpy() for h in hist]

    # plot 
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,num_epochs+1), hist1, label="Pretrained")
    plt.plot(range(1,num_epochs+1), hist2, label="Non-Pretrained")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.show()