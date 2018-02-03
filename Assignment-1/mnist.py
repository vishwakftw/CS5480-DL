from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt

# Training settings
batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.001

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

optimizer = optim.SGD(model.parameters(), lr=lr)


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

# One subplot for losses
plt.subplot(121)
plt.plot(range(1, epochs + 1), training_losses, 'r-', label='Training Losses')
plt.plot(range(1, epochs + 1), test_losses, 'b-', label='Testing Losses')
plt.title('Loss variation with Epochs')
plt.legend(loc='upper right')

# One subplot for accuracies
plt.subplot(122)
plt.plot(range(1, epochs + 1), training_accs, 'r-', label='Training Accuracy')
plt.plot(range(1, epochs + 1), test_accs, 'b-', label='Testing Accuracy')
plt.title('Accuracy variation with Epochs')
plt.legend(loc='lower right')

plt.show()
