import torch
from model import Model
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import os
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

def training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # dowload the dataset
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor(), download=False)
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor(), download= False)
    # select batch size
    batch_size = 256

    # dataloader creation
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    #print('train data : ', train_dataset)
    #print('test datat : ', test_dataset)
    #Model creation
    model = Model().to(device)
    sgd = SGD(model.parameters(), lr=1e-1) # stochastic gradient descent
    loss_fn = CrossEntropyLoss()
    all_epoch = 50
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
        torch.save(model, 'models/mnist_{:.3f}.pkl'.format(acc))
        if np.abs(acc - prev_acc) < 1e-4:
            break
        prev_acc = acc
    print("Model finished training")