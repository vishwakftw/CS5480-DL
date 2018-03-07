import torch as t
import numpy as np
import torch.nn as nn
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.autograd import Variable as V
from torchvision.transforms import Compose, ToTensor, Normalize

def weights_init(m, scheme='randn'):
    if scheme == 'uni':
        initializer = nn.init.uniform
    if scheme == 'randn':
        initializer = nn.init.normal
    elif scheme == 'xavier':
        initializer = nn.init.xavier_uniform
    elif scheme == 'kaiming':
        initializer = nn.init.kaiming_uniform

    if isinstance(m, nn.Conv2d):
        initializer(m.weight.data)
        initializer(m.bias.data)
    elif isinstance(m, nn.Linear):
        initializer(m.weight.data)
        initializer(m.bias.data)

class Net(nn.Module):
    def __init__(self, drop_prob=0.5, batch_norm=False, activation='relu', init_scheme='randn'):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout1d = nn.Dropout(drop_prob)
        self.dropout2d = nn.Dropout2d(drop_prob)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bnorm1 = nn.BatchNorm2d(10)
            self.bnorm2 = nn.BatchNorm2d(20)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh() 
        self.maxpool = nn.MaxPool2d(2)

        # Initialize the weights
        weights_init(self.conv1, init_scheme)
        weights_init(self.conv2, init_scheme)
        weights_init(self.fc1, init_scheme)
        weights_init(self.fc2, init_scheme)

    def forward(self, input):
        pass_ = self.maxpool(self.conv1(input))
        if self.batch_norm:
            pass_ = self.bnorm1(pass_)
        pass_ = self.activation(pass_)
        pass_ = self.maxpool(self.dropout2d(self.conv2(pass_)))
        if self.batch_norm:
            pass_ = self.bnorm2(pass_)
        pass_ = self.activation(pass_)
        pass_ = pass_.view(-1, 320)
        pass_ = self.dropout1d(self.activation(self.fc1(pass_)))
        return self.fc2(pass_)

p = ArgumentParser()
p.add_argument('--root', required=True, type=str, help='Root location for the dataset')
p.add_argument('--lr', default=1e-04, type=float, help='Learning rate for SGD')
p.add_argument('--batch_size', default=64, type=int, help='Batch Size')
p.add_argument('--activation', default='relu', type=str, help='Activation function')
p.add_argument('--dropout_prob', default=0.5, type=float, help='Dropout probability')
p.add_argument('--batch_norm', action='store_true', help='Toggle to include Batch Normalization')
p.add_argument('--init_scheme', default='randn', type=str, help='Initialization scheme:\nRandom Normal:\
                                                      \'randn\'\nXavier: \'xavier\'\nKaiming: \'kaiming\'')
p.add_argument('--epochs', default=10, type=int, help='Number of epochs to train for')
p = p.parse_args()

CUDA_CHECK = t.cuda.is_available()

# Get Dataset and DataLoader
transformations = [ToTensor(), Normalize((0.1307,), (0.3081,))]
tr_data = MNIST(root=p.root, train=True, download=True, transform=Compose(transformations))
te_data = MNIST(root=p.root, train=False, download=True, transform=Compose(transformations))

tr_loader = DataLoader(tr_data, batch_size=p.batch_size, shuffle=True)
te_loader = DataLoader(te_data, batch_size=1000, shuffle=True)

# Get Network, initialize. Get loss function and optimizer
model = Net(drop_prob=p.dropout_prob, batch_norm=p.batch_norm, activation=p.activation, init_scheme=p.init_scheme)
loss = nn.CrossEntropyLoss()

if CUDA_CHECK:
    model = model.cuda()
    loss = loss.cuda()

optimizer = t.optim.SGD(model.parameters(), lr=p.lr, weight_decay=1e-04)

# Build the training loop
n_iters = 0
train_losses = []
train_accus = []
test_losses = []
test_accus = []
for e in range(0, p.epochs):

    # Training phase
    model.train()
    for i, itr in enumerate(tr_loader):
        model.zero_grad()
        x, y = itr
        if CUDA_CHECK:
            x = x.cuda()
            y = y.cuda()
        x = V(x)
        y = V(y)
        outputs = model(x)
        cur_loss = loss(outputs, y)
        if n_iters % 500 == 0:
            print(round(cur_loss.data[0], 5))
        cur_loss.backward()
        optimizer.step()
        n_iters += 1

    # Evaluation phase
    model.eval()
    train_loss = 0.0
    train_accu = 0.0
    for i, itr in enumerate(tr_loader):
        x, y = itr
        if CUDA_CHECK:
            x = x.cuda()
            y = y.cuda()
        x = V(x)
        y = V(y)
        outputs = model(x)
        cur_loss = loss(outputs, y)
        train_loss += cur_loss.data[0] * len(y)
        pred_label = outputs.data.max(1)[1]
        train_accu += t.eq(pred_label, y.data).cpu().sum()

    train_loss /= len(tr_dataset)
    train_accu /= len(tr_dataset)
    train_losses.append(train_loss)
    train_accus.append(train_accu)

    test_loss = 0.0
    test_accu = 0.0
    for i, itr in enumerate(te_loader):
        x, y = itr
        if CUDA_CHECK:
            x = x.cuda()
            y = y.cuda()
        x = V(x)
        y = V(y)
        outputs = model(x)
        cur_loss = loss(outputs, y)
        test_loss += cur_loss.data[0] * len(y)
        pred_label = outputs.data.max(1)[1]
        test_accu += t.eq(pred_label, y.data).cpu().sum()

    test_loss /= len(te_dataset)
    test_accu /= len(te_dataset)
    test_losses.append(test_loss)
    test_accus.append(test_accu)
