import torch
from torch.utils.data import Dataset, DataLoader

class IntDataset(Dataset):

    def __init__(self, n, transform=None):
        self.data = range(n)  # lazy load data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = self.data[i]

        if self.transform:
            sample = self.transform(sample)

        return sample


class NegativeTransform(object):

    def __call__(self, sample):
        return -(sample)


class PowTransform(object):

    def __init__(self, power=2):
        self.power = power

    def __call__(self, sample):
        return sample ** self.power


class Compose(object):
    """compose many transforms."""

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


if __name__ == '__main__':

    dataset = IntDataset(10, Compose(
        NegativeTransform(),
        PowTransform(3)
    ))

    for sample in dataset:
        print(sample)


    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    for batch in dataloader:
        print(batch)