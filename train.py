import torch
from model import Model
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import os
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

import device
import metrics


def training():

    # dowload the dataset and normalize the input images
    trans = transforms.Compose([
        # Apply data aumentation to avoid overfitting
        transforms.RandAugment(2,9),
        # To resize image
        transforms.Resize((32, 32)),
        # transform array to tensor
        transforms.ToTensor(),
        # To normalize image
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.MNIST(
        root='./train',
        train=True,
        download=False,
        # apply transformation
        transform=trans
    )



    # this function will generate random indexes between 0 and 59999
    def split_indices(n, val_per, seed=0):
        n_val = int(n * val_per)
        np.random.seed(seed)
        idx = np.random.permutation(n)
        return idx[n_val:], idx[: n_val]


    val_per = 0.2
    rand_seed = 42
    # split the train dataset
    train_indices, val_indices = split_indices(len(train_set), val_per, rand_seed)

    # select batch size
    batch_size = 256
    model = Model(num_classes=10)

    # dataloader creation

    # choose random indices from the train_examples
    train_sampler = SubsetRandomSampler(train_indices)
    train_dl = DataLoader(train_set, batch_size, sampler=train_sampler)
    val_sampler = SubsetRandomSampler(val_indices)
    val_dl = DataLoader(train_set, batch_size, sampler=val_sampler)

    # use the default device for training (GPU or CPU)
    my_device = device.get_default_device()
    device.to_device(model, my_device)
    # push the dataloaders on the selected device
    train_dl = device.DeviceDataLoader(train_dl, my_device)
    val_dl = device.DeviceDataLoader(val_dl, my_device)

    num_epochs = 25

    # optimizer to updates weights of the model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # reduce the LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='max', verbose=True)

    # train the model
    history = metrics.fit(num_epochs, model, F.cross_entropy, train_dl, val_dl, optimizer, metrics.accuracy, scheduler, 'val_metric')

