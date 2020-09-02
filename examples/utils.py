import sys
sys.path.append('..')

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

from lrp import Sequential, Linear, Conv2d

def get_mnist_model():
    model = Sequential(
        Conv2d(1, 32, 3, 1, 1),
        nn.ReLU(),
        Conv2d(32, 32, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        Linear(14*14*32, 10)
    )
    return model

def get_mnist_data(batch_size=16):
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)

    test = torchvision.datasets.MNIST('../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    return train_loader, test_loader

def prepare_mnist_model(model, model_path="model.pth", epochs=1, lr=1e-3):
    if os.path.exists(model_path): 
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    else: 
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for e in range(epochs):
            for i, (x, y) in enumerate( train_loader):
                y_hat = model(x)
                loss  = loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                acc = (y == y_hat.max(1)[1]).float().sum() / x.size(0)
                if i%10 == 0: 
                    print("\r[%i/%i, %i/%i] loss: %.4f acc: %.4f" % (e, epochs, i, len(train_loader), loss.item(), acc.item()), end="", flush=True)
        torch.save(model.state_dict(), model_path)
