import torchvision.transforms as transforms
import torchvision
import device
import torch
import matplotlib.pyplot as plt
import model
import metrics
from torch.utils.data import DataLoader
import torch.nn.functional as F

trans = transforms.Compose([
    # To resize image
    #transforms.RandAugment(2, 9),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    # To normalize image
    transforms.Normalize((0.5,), (0.5,))
])

test_set = torchvision.datasets.MNIST(
    root='./test',
    train=False,
    download=False,
    transform=trans
)
Model = torch.load("/Users/arthurlamard/Documents/ISEN5/deep_learning/CNN/TP1/models/best_model.pt")
my_device = device.get_default_device()
def predict_image(img, model):
    xb = device.to_device(img.unsqueeze(0), my_device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()

img, label = test_set[0]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, Model))

test_loader = device.DeviceDataLoader(DataLoader(test_set, batch_size=256), my_device)
result = metrics.evaluate(Model, F.cross_entropy, test_loader, metric = metrics.accuracy)
#result
Accuracy = result[2] * 100
#Accuracy
loss = result[0]
print("Total Losses: {}, Accuracy: {}".format(loss, Accuracy))

