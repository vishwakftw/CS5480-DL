import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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
    elif isinstance(m, nn.Linear):
        initializer(m.weight.data)

class Net(nn.Module):
    def __init__(self, drop_prob=0.25, batch_norm=False, activation='relu', init_scheme=None):
        super(Net, self).__init__()
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=drop_prob)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'tanh':
            self.activation = F.tanh
        self.bnorm1 = nn.BatchNorm2d(10)
        self.bnorm2 = nn.BatchNorm2d(20)

        if init_scheme is not None:
            # Initialize the weights
            weights_init(self.conv1, init_scheme)
            weights_init(self.conv2, init_scheme)
            weights_init(self.fc1, init_scheme)
            weights_init(self.fc2, init_scheme)

    def forward(self, x):
        if self.batch_norm:
            x = self.activation(self.bnorm1(F.max_pool2d(self.conv1(x), 2)))
            x = self.activation(self.bnorm2(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)))
        else:
            x = self.activation(F.max_pool2d(self.conv1(x), 2))
            x = self.activation(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.activation(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.drop_prob)
        x = self.fc2(x)
        return F.log_softmax(x)

p = ArgumentParser()
p.add_argument('--root', required=True, type=str, help='Root location for the dataset')
p.add_argument('--lr', default=1e-04, type=float, help='Learning rate for SGD')
p.add_argument('--batch_size', default=64, type=int, help='Batch Size')
p.add_argument('--activation', default='relu', type=str, help='Activation function')
p.add_argument('--dropout_prob', default=0.25, type=float, help='Dropout probability')
p.add_argument('--batch_norm', action='store_true', help='Toggle to include Batch Normalization')
p.add_argument('--init_scheme', default=None, type=str, help='Initialization scheme:\nRandom Normal:\
                                                      \'randn\'\nXavier: \'xavier\'\nKaiming: \'kaiming\'')
p.add_argument('--epochs', default=5, type=int, help='Number of epochs to train for')
p = p.parse_args()

CUDA_CHECK = t.cuda.is_available()

# Get Dataset and DataLoader
transformations = [ToTensor(), Normalize((0.1307,), (0.3081,))]
tr_dataset = MNIST(root=p.root, train=True, download=True, transform=Compose(transformations))
te_dataset = MNIST(root=p.root, train=False, download=True, transform=Compose(transformations))

tr_loader = DataLoader(tr_dataset, batch_size=p.batch_size, shuffle=True)
te_loader = DataLoader(te_dataset, batch_size=1000, shuffle=True)

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
train_loss_log = np.empty(0)
for e in range(0, p.epochs):

    # Training phase
    model.train()
    for i, itr in enumerate(tr_loader):
        x, y = itr
        if CUDA_CHECK:
            x, y = x.cuda(), y.cuda()
        x, y = V(x), V(y)
        optimizer.zero_grad()
        outputs = model(x)
        cur_loss = F.nll_loss(outputs, y)
        train_loss_log = np.append(train_loss_log, round(cur_loss.data[0], 6))
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
            x, y = x.cuda(), y.cuda()
        x, y = V(x), V(y)
        outputs = model(x)
        cur_loss = F.nll_loss(outputs, y, size_average=False)
        train_loss += cur_loss.data[0]
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
            x, y = x.cuda(), y.cuda()
        x, y = V(x), V(y)
        outputs = model(x)
        cur_loss = F.nll_loss(outputs, y, size_average=False)
        test_loss += cur_loss.data[0]
        pred_label = outputs.data.max(1)[1]
        test_accu += t.eq(pred_label, y.data).cpu().sum()

    test_loss /= len(te_dataset)
    test_accu /= len(te_dataset)
    test_losses.append(test_loss)
    test_accus.append(test_accu)

    print("Epoch {} completed".format(e + 1))

plt.clf()
plt.figure(figsize=(12, 10))
plt.subplot(221)
plt.plot(list(range(1, p.epochs + 1)), train_losses, linewidth=2.0, markersize=5.0, marker='o', label='Train Loss')
plt.legend(loc='upper right')

plt.subplot(222)
plt.plot(list(range(1, p.epochs + 1)), train_accus, linewidth=2.0, markersize=5.0, marker='o', label='Train Accuracy')
plt.legend(loc='lower right')

plt.subplot(212)
plt.plot(list(range(n_iters)), train_loss_log, 'r-', linewidth=2.0, label='Training loss variation')
plt.legend(loc='upper right')
plt.savefig('Train-Statistics-MNIST-dropout-{}-batchnorm-{}-activation-{}-init-{}.png'.format(
            p.dropout_prob, p.batch_norm, p.activation, p.init_scheme), dpi=100)

plt.clf()
plt.figure(figsize=(12, 10))
plt.subplot(121)
plt.plot(list(range(1, p.epochs + 1)), test_losses, 'g-', linewidth=2.0, markersize=5.0, marker='o', label='Test Loss')
plt.legend(loc='upper right')

plt.subplot(122)
plt.plot(list(range(1, p.epochs + 1)), test_accus, 'g-', linewidth=2.0, markersize=5.0, marker='o', label='Test Accuracy')
plt.legend(loc='lower right')
plt.savefig('Test-Statistics-MNIST-dropout-{}-batchnorm-{}-activation-{}-init-{}.png'.format(
            p.dropout_prob, p.batch_norm, p.activation, p.init_scheme), dpi=100)
