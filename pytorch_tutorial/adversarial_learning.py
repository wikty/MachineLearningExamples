"""
Research is constantly pushing ML models to be faster, more accurate, 
and more efficient. However, an often overlooked aspect of designing
and training models is security and robustness, especially in the face
of an adversary who wishes to fool the model.

For context, there are many categories of adversarial attacks, each with
a different goal and assumption of the attacker’s knowledge. However, in
general the overarching goal is to add the least amount of perturbation to
the input data to cause the desired misclassification. 

* There are several kinds of assumptions of the attacker’s knowledge, two 
of which are: white-box and black-box. 

    - A white-box attack assumes the attacker has full knowledge and access 
    to the model, including architecture, inputs, outputs, and weights. 

    - A black-box attack assumes the attacker only has access to the inputs
    and outputs of the model, and knows nothing about the underlying 
    architecture or weights. 

* There are also several types of goals, including misclassification and
source/target misclassification. 

    - A goal of misclassification means the adversary only wants the output
    classification to be wrong but does not care what the new classification 
    is. 

    - A source/target misclassification means the adversary wants to alter an
    image that is originally of a specific source class so that it is 
    classified as a specific target class.

We will use one of the first and most popular attack methods, the Fast 
Gradient Sign Attack (FGSM), to fool an MNIST classifier. the FGSM attack
is a white-box attack with the goal of misclassification.

The attack is remarkably powerful, and yet intuitive. It is designed to attack
neural networks by leveraging the way they learn, gradients. The idea is 
simple, rather than working to minimize the loss by adjusting the weights 
based on the backpropagated gradients, the attack adjusts the input data to 
maximize the loss based on the same backpropagated gradients. In other words, 
the attack uses the gradient of the loss w.r.t the input data, then adjusts 
the input data to maximize the loss.

More: 
- https://arxiv.org/abs/1412.6572
- https://arxiv.org/pdf/1804.00097.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


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


def load_test_data(data_dir, shuffle=True, num_workers=2):
    # a sequential transforms for PIL Image
    transform = transforms.Compose([
        # Convert PIL Image (HxWxC) in range [0, 255] into
        # tensor (CxHxW) in range [0.0, 1.0].
        transforms.ToTensor(),
        # Normalization: change the range from [0.0, 1.0] to
        # [-1.0, 1.0]
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    testset = torchvision.datasets.MNIST(root=data_dir, train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=1,
                                             shuffle=shuffle, 
                                             num_workers=num_workers)
    return testloader


def load_pretrained_model(model_path, training=False, device=None):
    model = Net()
    # load and remap model state to cpu device
    model_state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_state)
    if training:
        model.train()
    else:
        model.eval()
    if device:
        model = model.to(device)
    return model


def fgsm_attack(image, epsilon):
    """Create the adversarial example by perturbing the original image.

    Params:
        image: the image tensor must be tracked gradient.
        epsilon: the strength of adversary between [0, 1)
    """
    # the gradient of the loss w.r.t the image
    grad = image.grad.data
    # get the sign of gradient
    sign_grad = grad.sign()
    # create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_grad  # gradient ascent
    # clipping to [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def test(model, dataloader, criterion, epsilon, device, num_examples=5):
    correct = 0
    adv_examples = []

    for image, label in dataloader:
        image, label = image.to(device), label.to(device)
        image.requires_grad_()  # track gradient for the input image
        
        # forward: calculate the output of the input image
        output = model(image)
        init_pred = output.max(dim=1)[1]

        # pretrained model make a wrong prediction
        if init_pred.item() != label.item():
            continue

        # backward: calculate the gradient of the input image
        model.zero_grad()
        loss = criterion(output, label)
        loss.backward()

        # use FGSM to perturbe the image
        perturbed_image = fgsm_attack(image, epsilon)
        
        # check the perturbed image
        output = model(perturbed_image)
        final_pred = output.max(dim=1)[1]

        if final_pred.item() == label.item():
            correct += 1
        else:
            adv_examples.append((
                init_pred.item(), 
                final_pred.item(),
                perturbed_image.squeeze().detach().cpu().numpy()))

    acc = float(correct)/len(dataloader)

    return acc, adv_examples[:num_examples]




if __name__ == '__main__':
    data_dir = '../data/MNIST'
    model_path = '../models/MNIST/lenet_mnist_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # The strength of adversary:
    # - 0 means that don't perturbe image
    # - the larger of epsilon, the more effective the attack
    # - because image pixel range in [0, 1], epsilon larger than 1
    # has no effect(will be clamp to 1).
    epsilons = [0, .05, .1, .15, .2, .25, .3]

    num_examples = 5  # return the number of adversary examples
    accuracies = []
    examples = []

    dataloader = load_test_data(data_dir)
    model = load_pretrained_model(model_path, False, device)
    criterion = F.nll_loss

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, dataloader, criterion, eps, device, 
                       num_examples)
        print('Epsilon: {}, Test Acc: {}'.format(eps, acc))
        accuracies.append(acc)
        examples.append(ex)

    # as epsilon increases we expect the test accuracy to decrease.
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    # as epsilon increases the test accuracy decreases BUT the 
    # perturbations become more easily perceptible.
    rows, columns = len(epsilons), num_examples
    cnt = 0
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(rows, columns, cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()
