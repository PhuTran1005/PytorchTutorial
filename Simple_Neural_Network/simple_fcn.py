import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# create Fully Connected Network
class FCN(nn.Module):
    def __init__(self, input_size, hidden_dim=50, num_classes=10):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F_torch.relu(x)
        x = self.fc2(x)

        return x
    

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# hyperparameters setting
input_size = 28 * 28 # width and height of MNIST dataset
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 20

# load data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="datatset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# initialize model
model = FCN(input_size=input_size, num_classes=num_classes).to(device=device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training network
for epoch in range(num_epochs):
    for batch_idx, (images, targets) in enumerate(train_loader):
        # load data to GPU if posible
        images = images.to(device=device)
        targets = targets.to(device=device)

        # reshape input images to reshape of model
        images = images.reshape(images.shape[0], -1)

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