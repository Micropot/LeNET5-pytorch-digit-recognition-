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
    '''device = 'cpu'
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else: device = 'cpu''
    print("DEVICE USED : ", device)'''

    # dowload the dataset and normalize the input images
    trans = transforms.Compose([
        # To resize image
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # To normalize image
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.MNIST(
        root='./train',
        train=True,
        download=False,
        transform=trans
    )

    test_set = torchvision.datasets.MNIST(
        root='./test',
        train=False,
        download=False,
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

    train_indices, val_indices = split_indices(len(train_set), val_per, rand_seed)

    # select batch size
    batch_size = 256
    #model = Model()
    model = Model(num_classes=10)

    # dataloader creation
    train_sampler = SubsetRandomSampler(train_indices)
    train_dl = DataLoader(train_set, batch_size, sampler=train_sampler)
    val_sampler = SubsetRandomSampler(val_indices)
    val_dl = DataLoader(train_set, batch_size, sampler=val_sampler)

    my_device = device.get_default_device()
    device.to_device(model, my_device)
    print('device : ', my_device)

    train_dl = device.DeviceDataLoader(train_dl, my_device)
    val_dl = device.DeviceDataLoader(val_dl, my_device)

    num_epochs = 25

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='max', verbose=True)

    history = metrics.fit(num_epochs, model, F.cross_entropy, train_dl, val_dl, optimizer, metrics.accuracy, scheduler, 'val_metric')

    '''#print('train data : ', train_dataset)
    #print('test datat : ', test_dataset)
    #Model creation
    model = Model().to(device)
    sgd = SGD(model.parameters(), lr=1e-1) # stochastic gradient descent
    loss_fn = CrossEntropyLoss()
    all_epoch = 100
    prev_acc = 0
    for current_epoch in range(all_epoch): # epoch by epoch
        model.train() # train model using the class model defined (call forward)
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            sgd.zero_grad() # zero out the gradient to update the parameters correctly
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long()) # calculate the loss
            loss.backward() # backward propagration
            sgd.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        # evaluate the model on the test dataset
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            print("test_x.shape : ",test_x.shape)
            predict_y = model(test_x.float()).detach()
            predict_y = torch.argmax(predict_y, dim=-1) # maximum value of the input tensor
            current_correct_num = predict_y == test_label
            #print("current correct num : ", current_correct_num)
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            #print('current_correct_num.shape[0] : ', current_correct_num.shape[0])
            all_sample_num += current_correct_num.shape[0]
        print('all correct num : ', all_correct_num)
        print('all sample num: ', all_sample_num)
        acc = all_correct_num / all_sample_num
        #print('current epoch : ', current_epoch)
        print('Epoch[{}/{}], accuracy: {:.3f}'.format(current_epoch+1, all_epoch, acc), flush=True)
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(model, 'models/mnist_{:.3f}.pt'.format(acc))
        if np.abs(acc - prev_acc) < 1e-4:
            break
        prev_acc = acc
    print("Model finished training")'''