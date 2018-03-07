import torch as t
import numpy as np
import torch.nn as nn
from torch.autograd import Variable as V

class Net(nn.Module):
    def __init__(self, drop_prob=(0.5, 0.5), batch_norm=False, activation='relu'):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout1d = nn.Dropout(drop_prob[0])
        self.dropout2d = nn.Dropout2d(drop_prob[1])
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
