import torch
import torch.nn as nn # all neural networks modules: nn.Linear, nn.Conv2d, BatchNorm, Loss Func
import torch.optim as optim # optimization algorithms: Adam, SGD, ...
import torch.nn.functional as F_torch # all functions that don't have any parameters
from torch.utils.data import DataLoader # gives easier dataset management and creates mini batches dataset
import torchvision.datasets as datasets # include standard datasets we can import in a nice way
import torchvision.transforms as transforms # transforms datasets


# create a simple Convolution Neural Network
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F_torch.relu(self.conv1(x))
        x = self.pool(x)
        x = F_torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# hyuperparameters
in_channels=1
num_classes=10
learning_rate=1e-3
batch_size = 64
num_epochs = 20

# load data
train_datasets = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True)

test_datasets = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=False)

# initialize network
model = CNN().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training network
for epoch in range(num_epochs):
    for batch_idx, (images, targets) in enumerate(train_loader):
        # load data to GPU if posible
        images = images.to(device=device)
        targets = targets.to(device=device)

        # forward part
        preds = model(images) # predicted images
        loss = criterion(preds, targets) # calc loss

        # backward part
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# check accuracy during training and testing to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking acc on train dataset')
    else:
        print('Checking acc on test dataset')

    num_correct = 0
    num_samples = 0
    model.eval() # evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).reshape(x.shape[0], -1)
            y = y.to(device=device)
            
            preds = model(x)
            _, predictions = preds.max(1)
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy is {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()
    

if __name__ == '__main__':
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)
