import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def save_checkpoint(state, file_name="my_checkpoint.pth.tar"):
    print('=> Saving checkpoint...')
    torch.save(state, file_name)
    print('Finish saving checkpoint.')


def load_checkpoint(checkpoint, model, optimizer):
    print('=> Loading checkpoint...')
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print('Finish loading checkpoint.')


def main():
    # initialize network
    model = torchvision.models.vgg16(
        weights=None
    ) # pretrained = False deprecatedm use weights instead
    optimizer = optim.Adam(
        model.parameters()
    )
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    # calling save checkpoint function
    save_checkpoint(checkpoint)

    # calling load checkpoint function
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


if __name__ == '__main__':
    main()