import sys
import pathlib
# Append parent directory of this file to sys.path, 
# no matter where it is run from
base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())

import pickle
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

from lrp import Sequential, Linear, Conv2d, MaxPool2d

_standard_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# # # # # # # # # # # # # # # # # # # # # 
# MNIST
# # # # # # # # # # # # # # # # # # # # # 
def get_mnist_model():
    model = Sequential(
        Conv2d(1, 32, 3, 1, 1),
        nn.ReLU(),
        Conv2d(32, 64, 3, 1, 1),
        nn.ReLU(),
        MaxPool2d(2,2),
        nn.Flatten(),
        Linear(14*14*64, 512),
        nn.ReLU(),
        Linear(512, 10)
    )
    return model

def get_mnist_data(transform, batch_size=32):
    train = torchvision.datasets.MNIST((base_path / 'data').as_posix(), train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    test = torchvision.datasets.MNIST((base_path / 'data').as_posix(), train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def prepare_mnist_model(args, model, model_path=(base_path / 'examples' / 'models' / 'mnist_model.pth').as_posix(), epochs=1, lr=1e-3, train_new=False, transform=_standard_transform):
    train_loader, test_loader = get_mnist_data(transform)

    if os.path.exists(model_path) and not train_new: 
        state_dict = torch.load(model_path, map_location=args.device)
        model.load_state_dict(state_dict)
    else: 
        device = args.device
        model = model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for e in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                loss  = loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                acc = (y == y_hat.max(1)[1]).float().sum() / x.size(0)
                if i%10 == 0: 
                    print("\r[%i/%i, %i/%i] loss: %.4f acc: %.4f" % (e, epochs, i, len(train_loader), loss.item(), acc.item()), end="", flush=True)
        torch.save(model.state_dict(), model_path)

# # # # # # # # # # # # # # # # # # # # # 
# Patterns
# # # # # # # # # # # # # # # # # # # # # 
def store_patterns(file_name, patterns):
    with open(file_name, 'wb') as f:
        pickle.dump([p.detach().cpu().numpy() for p in patterns], f)

def load_patterns(file_name): 
    with open(file_name, 'rb') as f: p = pickle.load(f)
    return p

