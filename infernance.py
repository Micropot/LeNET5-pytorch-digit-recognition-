import torchvision.transforms as transforms
import torchvision
import device
import torch
import metrics
from torch.utils.data import DataLoader
import torch.nn.functional as F


# transformations for the test base
trans = transforms.Compose([
    # To resize image
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


# fucntion to predict the input image using the trained model
def predict_image(img, model):

    my_device = device.get_default_device()
    xb = device.to_device(img.unsqueeze(0), my_device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()


my_device = device.get_default_device()
Model = torch.load("/Users/arthurlamard/Documents/ISEN5/deep_learning/CNN/TP1/models/best_model.pt")
test_loader = device.DeviceDataLoader(DataLoader(test_set, batch_size=256), my_device)
result = metrics.evaluate(Model, F.cross_entropy, test_loader, metric = metrics.accuracy)
#result
Accuracy = result[2] * 100
#Accuracy
loss = result[0]
print("Total Losses: {}, Accuracy: {}".format(loss, Accuracy))
