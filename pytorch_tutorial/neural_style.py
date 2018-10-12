"""
Neural-Style, or Neural-Transfer, allows you to take an image and reproduce it
with a new artistic style. The algorithm takes three images, an input image, a
content-image, and a style-image, and changes the input to resemble the 
content of the content-image and the artistic style of the style-image.

The principle is simple: we define two distances, one for the content (DC) and
one for the style (DS). DC measures how different the content is between two 
images while DS measures how different the style is between two images. Then, 
we take a third image, the input, and transform it to minimize both its 
content-distance with the content-image and its style-distance with the 
style-image. Now we can import the necessary packages and begin the neural 
transfer.

More: https://arxiv.org/abs/1508.06576
"""
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models


def load_img(path, imsize, device):
    """
    The original PIL images have values between 0 and 255, but when transformed into torch tensors, their values are converted to be between 0 and 1. The images also need to be resized to have the same dimensions. An important detail to note is that neural networks from the torch library are trained with tensor values ranging from 0 to 1. If you try to feed the networks with 0 to 255 tensor images, then the activated feature maps will be unable sense the intended content and style. However, pre-trained networks from the Caffe library are trained with 0 to 255 tensor images.
    """
    transform = transforms.Compose([
        transforms.Resize(imsize),  # resize image
        transforms.ToTensor()  # PIL image to Tensor
    ])
    img = Image.open(path)
    #  fake batch dimension required to fit network's input dimensions
    img = transform(img).unsqueeze(0)
    return img.to(device, torch.float)


def show_img(tensor, title=None):
    transform = transforms.ToPILImage()  # Tensor to PIL image
    tensor = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    tensor = tensor.squeeze(0)  # remove the fake batch dimension
    img = transform(tensor)

    plt.imshow(img)
    if title is not None:
        plt.title(title)

    plt.pause(0.001) # pause a bit so that plots are updated


class Normalization(nn.Module):
    """The module to normalize the input image. We will add it as the first
    layer in our model."""

    def __init__(self, mean, std):
        super().__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):
    """The Module calculates the content loss for for an individual layer.
    We will add it after each desired convolutional layer.
    """

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        """Don't change input and transparent calculation the loss."""
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    """The Module calculates the style loss for for an individual layer.
    We will add it after each desired convolutional layer.
    """

    def __init__(self, target):
        super().__init__()
        self.target = self.gram_matrix(target.detach())

    def gram_matrix(self, batch):
        # a=batch size
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        a, b, c, d = batch.size()

        features = batch.view(a * b, c * d)

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def forward(self, input):
        """Don't change input and transparent calculation the loss."""
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


def create_model(content_img, style_img, content_loss_layers, 
                 style_loss_layers, device):
    """
    Params:
        content_img: the tensor of content image.
        style_img: the tensor of style image.
        content_loss_layers: the desired layers to be followed ContentLoss
            layer.
        style_loss_layers: the desired layers to be followed StyleLoss layer.
        device: the device to run model.
    """
    #
    # Import a pretrained model
    # -------------------------
    # Now we need to import a pre-trained neural network. We will use a 19
    # layer VGG network like the one used in the paper.
    # 
    # PyTorch’s implementation of VGG is a module divided into two child
    # ``Sequential`` modules: ``features`` (containing convolution and pooling layers),
    # and ``classifier`` (containing fully connected layers). We will use the
    # ``features`` module because we need the output of the individual
    # convolution layers to measure content and style loss. Some layers have
    # different behavior during training than evaluation, so we must set the
    # network to evaluation mode using ``.eval()``.
    # 

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # Additionally, VGG networks are trained on images with each channel
    # normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    # We will use them to normalize the image before sending it into the network.
    # 

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization = Normalization(mean, std).to(device)

    # Our model is a Sequential instance. 
    # Add the normalization module first.
    model = nn.Sequential(normalization)

    # store the list of content/style losses
    content_losses = []
    style_losses = []

    # Assuming that cnn is a nn.Sequential, iterate its children modules and
    # add into our model.
    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        # add the cnn layer module into our model
        model.add_module(name, layer)

        # add content loss module into our model
        if name in content_loss_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            content_losses.append(content_loss)
            model.add_module("content_loss_{}".format(i), content_loss)

        # add style loss module into our model
        if name in style_loss_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            style_losses.append(style_loss)
            model.add_module("style_loss_{}".format(i), style_loss)

    # remove layers after the last content and style losses.
    # those layers don't affect style transfer.
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, content_losses, style_losses



def run_style_transfer(model, content_losses, style_losses, 
                       content_img, style_img, num_steps=300, 
                       style_weight=1000000, content_weight=1):
    # Create the input image
    # Use the copy of the content image or white noise
    # input_img = torch.randn(cimg.data.size(), device=device)
    input_img = content_img.clone()
    original_img = input_img.clone()

    # Create optimizer
    # Note: the input image is a learnable parameter.
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    for i in range(num_steps):
        # LBFGS need to reevaluate the function multiple times, so you have
        # to pass in a closure that allows them to recompute your model. 
        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            print('Content-Loss: {}, Style-Loss: {}'.format(
                content_score.item(), style_score.item()))

            return loss

        optimizer.step(closure)

    # correct image
    input_img.data.clamp_(0, 1)

    return original_img, input_img


if __name__ == '__main__':
    """
    Command-Line Arguments:
    content image filename[optional]
    style image filename[optional]
    """

    content_img_path = '../data/neural_style/dancing.jpg'
    style_img_path = '../data/neural_style/picasso.jpg'

    if len(sys.argv) == 2:
        content_img_path = '../data/neural_style/{}'.format(sys.argv[1])
    elif len(sys.argv) == 3:
        content_img_path = '../data/neural_style/{}'.format(sys.argv[1])
        style_img_path = '../data/neural_style/{}'.format(sys.argv[2])

    # Running the neural transfer algorithm on large images takes longer and
    # will go much faster when running on a GPU. 
    imsize = 128
    device = torch.device('cpu')
    if torch.cuda.is_available():
        imsize = 512  # use large image size if GPU is avaiable
        device = torch.device('cuda')

    content_img = load_img(content_img_path, imsize, device)
    style_img = load_img(style_img_path, imsize, device)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    plt.ion()

    plt.figure()
    show_img(content_img, title="Content Image")
    plt.figure()
    show_img(style_img, title="Style Image")

    # display_images(content_img, style_img, imsize, device)

    # The desired convolution layers to add the ContentLoss and StyleLoss
    # layers immediately.
    content_loss_layers = ['conv_4']
    style_loss_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']   

    # Create the model
    model, content_losses, style_losses = create_model(
        content_img, style_img, content_loss_layers, 
        style_loss_layers, device)

    # Run style transfer
    input_img, output_img = run_style_transfer(model, content_losses, 
        style_losses, content_img, style_img, num_steps=100)

    # Show results
    plt.figure()
    show_img(input_img, title='Input Image')
    plt.figure()
    show_img(output_img, title='Output Image')
    plt.ioff()
    plt.show()