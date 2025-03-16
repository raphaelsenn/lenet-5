"""
Author: Raphael Senn
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from lenet_5.lenet_5 import LeNet5


def evaluate(model: nn.Module, dataloader: DataLoader, epochs: int=1, lr: float=0.01, verbose: bool=True) -> None:
    criterion = nn.CrossEntropyLoss()
    model.eval()


def train(model: nn.Module, dataloader: DataLoader, epochs: int=1, lr: float=0.01, verbose: bool=True) -> None:

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr) 

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_acc = 0.0
        for X, y in dataloader:
            # reset gradients
            optimizer.zero_grad() 

            # make predictions
            pred = model.forward(X)

            # calculate cross-entropy loss
            loss = criterion(pred, y)

            # backpropagation
            loss.backward()

            # update parameters
            optimizer.step()

            # cache loss and number of correct classifications
            total_loss += loss.item()
            total_acc += sum(torch.argmax(pred, dim=1) == y)

        if verbose:
            total_loss = total_loss / len(dataloader.dataset)
            total_acc = total_acc / len(dataloader.dataset)
            print(f'epoch: {epoch}\tloss: {total_loss}\tacc: {total_acc}')



if __name__ == '__main__':
    transform = transforms.Compose([torchvision.transforms.ToTensor()])
    minst_train = torchvision.datasets.MNIST(root='data/', transform=transform,download=True, train=True)
    minst_test = torchvision.datasets.MNIST(root='data/', transform=transform,download=True, train=False)

    dataloader_train = DataLoader(minst_train, batch_size=128)
    dataloader_test = DataLoader(minst_test, batch_size=128)

    lenet5 = LeNet5()
    train(lenet5, dataloader_train, epochs=30)

