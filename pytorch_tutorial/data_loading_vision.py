import os
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# skimage for image io and transforms
from skimage import io, transform
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def load_landmarks(csv_file):
    """load csv data as numpy array."""
    df = pd.read_csv(csv_file)
    filenames = []
    landmarks_list = []
    for index, row in df.iterrows():
        filename = df.iloc[index, 0]
        filenames.append(filename)
        landmarks = df.iloc[index, 1:].as_matrix()
        landmarks_list.append(landmarks.astype('float').reshape(-1, 2))
    return filenames, landmarks_list


def imshow(image, landmarks, pause=0.01):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', color='r')
    # plt.pause(pause)


def load_and_show_image(landmarks_csv, img_dir):
    filenames, landmarks_list = load_landmarks(landmarks_csv)
    plt.figure()
    i = random.randint(0, len(filenames)-1)
    img_file = os.path.join(img_dir, filenames[i])
    landmarks = landmarks_list[i]
    imshow(io.imread(img_file), landmarks, 3)
    plt.show()


class FaceLandmarksDataset(Dataset):
    """Face Landmarks Dataset."""

    def __init__(self, landmarks_csv, img_dir, transform=None):
        self.landmarks_frame = pd.read_csv(landmarks_csv)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """Return length of Dataset."""
        return len(self.landmarks_frame)

    def __getitem__(self, i):
        """Return a dict as a sample."""
        img_file = os.path.join(self.img_dir,
                                self.landmarks_frame.iloc[i, 0])
        image = io.imread(img_file)
        landmarks = self.landmarks_frame.iloc[i, 1:].as_matrix()  # all points
        landmarks = landmarks.astype('float').reshape(-1, 2)  # a point per row

        sample = {
            'image': image,
            'landmarks': landmarks
        }

        # apply transform for sample
        if self.transform:
            sample = self.transform(sample)

        return sample


def load_and_show_dataset(landmarks_csv, img_dir, transform=None):
    fig = plt.figure()
    dataset = FaceLandmarksDataset(landmarks_csv, img_dir, transform)
    n = 4  # show n images
    for i in range(len(dataset)):
        sample = dataset[i]
        if i > 3:
            plt.show()
            break
        
        ax = plt.subplot(1, n, i+1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        imshow(**sample)


class RescaleTransform(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple, int))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h < w:
                new_h = self.output_size
                new_w = new_h * (w / h)
            else:
                new_w = self.output_size
                new_h = new_w * (h / w)
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        # landmarks[:, 0] *= (new_w / w)  # first-dim is x
        # landmarks[:, 1] *= (new_h /h)  # second-dim is y
        landmarks = landmarks * [(new_w / w), (new_h / h)]

        return {'image': image, 'landmarks': landmarks}


class RandomCropTransform(object):
    """Crop randomly the image in a sample. This is data augmentation.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple, int))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top, left = np.random.randint(0, h - new_h), random.randint(0, w - new_w)

        image = image[top:top+new_h, left:left+new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensorTransform(object):
    """Convert NumPy ndarrays in sample to PyTorch Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        # ndarray to tensor
        image = torch.from_numpy(image)
        landmarks = torch.from_numpy(landmarks)

        return {'image': image, 'landmarks': landmarks}


def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    print('batch size', images_batch.size())
    im_size = images_batch.size(2)  # size: (batch_size, channel, im_h, im_w)
    # show a grid of images
    grid = utils.make_grid(images_batch)
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')
        plt.title('Batch from dataloader')


def load_data_with_torchvision(root_dir):
    """
    Maybe there is not necessary to define your dataset and transform class.
    torchvision package provides some common datasets and transforms.
    One of the more generic datasets available in torchvision is ImageFolder. 
    It assumes that images are organized in the following way:

        root/ants/xxx.png
        root/ants/xxy.jpeg
        root/ants/xxz.png
        .
        .
        .
        root/bees/123.jpg
        root/bees/nsdf3.png
        root/bees/asd932_.png

    where ‘ants’, ‘bees’ etc. are class labels.

    Similarly generic transforms which operate on PIL.Image like 
    RandomHorizontalFlip, Scale, are also avaiable. 
    """
    data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=root_dir,
                                   transform=data_transform)
    dataset_loader = DataLoader(dataset,
                                batch_size=4, shuffle=True,
                                num_workers=4)
    return dataset_loader



if __name__ == "__main__":

    landmarks_csv = '../data/faces/face_landmarks.csv'
    img_dir = '../data/faces'
    hymenoptera_dir = '../data/hymenoptera_data/train'



    # plt.ion()   # interactive mode
    # plt.ioff()  # non-interactive mode

    ## Show Image
    # load_and_show_image(landmarks_csv, img_dir)

    ## Show Transformed Image
    # plt.ioff()
    # composed = transforms.Compose([
    #     RescaleTransform(256),
    #     RandomCropTransform(255)
    # ])
    # load_and_show_dataset(landmarks_csv, img_dir, composed)

    # dataset
    dataset = FaceLandmarksDataset(landmarks_csv, img_dir, transforms.Compose([
        RescaleTransform(256),
        RandomCropTransform(224),
        ToTensorTransform()
    ]))

    # For-In dataset
    # - Load and transform a sample on fly
    for i in range(len(dataset)):
        sample = dataset[i]
        image, landmarks = sample['image'], sample['landmarks']
        print('Image: {}, Landmarks: {}'.format(image.size(), landmarks.size()))
        break
    
    # Iteration dataset
    # - Batching the data
    # - Shuffling the data
    # - Load the data in parallel using multiprocessing workers.
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, 
                            num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['landmarks'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            # plt.ioff()
            plt.show()
            break
    