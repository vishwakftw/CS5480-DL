from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
from argparse import ArgumentParser as AP

p = AP()
p.add_argument('--opt_name', type=str, required=True, help='name of the optimizer')
p.add_argument('--lr', type=float, required=True, help='learning rate')
p.add_argument('--momentum', type=float, default=0.0, help='momentum parameter')
p.add_argument('--l2', type=float, default=0.0, help='weight decay / L2 regularization')
p.add_argument('--nesterov', type=bool, default=False, help='nesterov momentum or not')
p.add_argument('--batchsize', type=int, required=True, help='mini-batch size')
p.add_argument('--beta1', type=float, default=0.9, help='beta 1 for ADAM')
p.add_argument('--beta2', type=float, default=0.99, help='beta 2 for ADAM')
p = p.parse_args()

# Training settings
test_batch_size = 1000
epochs = 10
lr = p.lr
momentum = p.momentum
weight_decay = p.l2
nesterov = p.nesterov
batch_size = p.batchsize
betas = (p.beta1, p.beta2)

seed = 1
log_interval = 10

torch.manual_seed(seed)
CUDA_CHECK = torch.cuda.is_available()
if CUDA_CHECK:
    torch.cuda.manual_seed(seed)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if CUDA_CHECK:
    model = model.cuda()

if p.opt_name == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
elif p.opt_name == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
elif p.opt_name == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if CUDA_CHECK:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()

    # First calculate Training loss over the entire dataset and then the Training accuracy
    train_loss = 0.0
    train_accu = 0.0
    for _, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if CUDA_CHECK:
            data = data.cuda()
            target = target.cuda()
        output = model(data)
        loss = F.nll_loss(output, target, size_average=False)
        train_loss += loss.data[0]
        pred_label = output.data.max(1)[1]
        train_accu += torch.eq(pred_label, target.data).cpu().sum()
    train_loss /= len(train_loader.dataset)
    train_accu /= len(train_loader.dataset)

    # Then calculate Testing loss over the entire dataset and then the Testing accuracy
    test_loss = 0.0
    test_accu = 0.0
    for _, (data, target) in enumerate(test_loader):
        data, target = Variable(data), Variable(target)
        if CUDA_CHECK:
            data = data.cuda()
            target = target.cuda()
        output = model(data)
        loss = F.nll_loss(output, target, size_average=False)
        test_loss += loss.data[0]
        pred_label = output.data.max(1)[1]
        test_accu += torch.eq(pred_label, target.data).cpu().sum()
    test_loss /= len(test_loader.dataset)
    test_accu /= len(test_loader.dataset)

    return train_loss, train_accu, test_loss, test_accu

training_losses = []
training_accs = []
test_losses = []
test_accs = []

for epoch in range(1, epochs + 1):
    train(epoch)
    tr_ls, tr_ac, te_ls, te_ac = test()
    print("Epoch {} completed".format(epoch))

    training_losses.append(tr_ls)
    training_accs.append(tr_ac)
    test_losses.append(te_ls)
    test_accs.append(te_ac)

fig = plt.figure(figsize=(12, 10))

# One subplot for losses
plt.subplot(121)
plt.plot(range(1, epochs + 1), training_losses, color='red', ls='-', marker='.', ms=12.5, label='Training Losses')
plt.plot(range(1, epochs + 1), test_losses,  color='blue', ls='-', marker='.', ms=12.5, label='Testing Losses')
plt.title('Loss variation with Epochs')
plt.legend(loc='upper right')

# One subplot for accuracies
plt.subplot(122)
plt.plot(range(1, epochs + 1), training_accs,  color='red', ls='-', marker='.', ms=12.5, label='Training Accuracy')
plt.plot(range(1, epochs + 1), test_accs,  color='blue', ls='-', marker='.', ms=12.5, label='Testing Accuracy')
plt.title('Accuracy variation with Epochs')
plt.legend(loc='lower right')

plt.savefig('{0}-{1}-{2}-{3}-{4}.png'.format(lr, momentum, weight_decay, nesterov, batch_size), dpi=100)
