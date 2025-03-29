"""
Author: Raphael Senn
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from lenet_5.lenet_5 import LeNet5


def evaluate(
        model: nn.Module,
        dataloader: DataLoader) -> None:
    
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    total_loss = 0.0
    total_acc = 0.0
    correct = 0
    miss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model.forward(X)
            loss = criterion(pred, y)   # calculates mean loss of the entire batch!

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_acc += (torch.argmax(pred, dim=1) == y).sum().item()
            correct += (torch.argmax(pred, dim=1) == y).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_acc / len(dataloader.dataset)
    mcr = 1 - accuracy # missclassification rate
    miss = len(dataloader.dataset) - correct  
    return avg_loss, accuracy, mcr, correct, miss


def train(
        model: nn.Module,
        dataloader: DataLoader,
        epochs: int=1,
        lr: float=0.1,
        verbose: bool=True) -> None:

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr) 

    model.train()
    for epoch in range(epochs):
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

        if verbose:
            avg_loss, acc, mcr, correct, miss = evaluate(model, dataloader)
            print(f'(train set)\tepoch: {epoch}\tloss: {avg_loss:.04f}\tacc: {acc:.04f}\tmcr: {mcr:.04f}\tcorrect: {correct}\tmiss: {miss}')


if __name__ == '__main__':
    seed = 42 
    torch.manual_seed(seed)
    transform = transforms.Compose([torchvision.transforms.ToTensor()])
    minst_train = torchvision.datasets.MNIST(root='data/', transform=transform,download=True, train=True)
    minst_test = torchvision.datasets.MNIST(root='data/', transform=transform,download=True, train=False)

    dataloader_train = DataLoader(minst_train, shuffle=True, batch_size=64)
    dataloader_test = DataLoader(minst_test, batch_size=64)

    lenet5 = LeNet5()
    train(lenet5, dataloader_train, epochs=20, lr=0.001)

    avg_loss, acc, mcr, correct, miss = evaluate(lenet5, dataloader_test)
    print(f'(test set)\tloss: {avg_loss:.04f}\tacc: {acc:.04f}\tmcr: {mcr:.04f}\tcorrect: {correct}\tmiss: {miss}')