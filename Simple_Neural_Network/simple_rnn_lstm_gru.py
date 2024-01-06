import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters setting
input_size = 28
sequence_length = 28
num_layers = 2
num_classes = 10
hidden_size = 256
learning_rate = 1e-3
batch_size = 64
num_epochs = 2

# create a RNN network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # RNN network
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # GRU 
        # self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        # LSTM network
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # hidden state initialize
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # only for lstm
        c0 = torch.zeros(self.num_layers, x_size(0), self.hidden_size).to(device)

        # forward pass
        # rnn
        # out, _ = self.rnn(x, h0)
        # gru
        # out, _ = self.gru(x, h0)
        # lstm
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return out

# load data
train_data = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# initialize network
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # get data to cuda if cuda is available
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # forward pass
        preds = model(data)
        loss = criterion(preds, targets)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# check accuracy on training and testing datasets to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training dataset...')
    else:
        print('Checking accuracy on testing dataset...')
    
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device=device).squeeze(1)
            targets = targets.to(device=device)

            preds = model(inputs)
            _, predictions = preds.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy \
            {float(num_correct) / float(num_samples)*100:.2f}')

    model.train()

if __name__ == '__main__':
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)