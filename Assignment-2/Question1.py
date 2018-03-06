import yaml
import torch as t
import pandas as pd
import torch.nn as nn
from argparse import ArgumentParser as AP
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

CUDA_CHECK = t.cuda.is_available()

p = AP()
p.add_argument('--train_data_loc', type=str, required=True, help='Root location for the training data')
p.add_argument('--test_data_loc', type=str, required=True, help='Root location for the testing data')
p.add_argument('--batch_size', type=int, default=100, help='Batch size for the training')
p.add_argument('--opt_det', type=str, required=True, help='Root location for the Optimizer details stored as YAML')
p.add_argument('--layer_info', type=str, required=True, help='Root location for CSV file with MLP layer information')
p.add_argument('--activation', type=str, default='relu', help='Activation for hidden layers')
p.add_argument('--loss', type=str, default='bce', help='Loss function')
p.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for')
p = p.parse_args()

# Read the dataset from the file
usecols = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Occupancy']
train_data = pd.read_csv(p.train_data_loc, usecols=usecols, sep=',')
train_data = train_data.values

test_data = pd.read_csv(p.test_data_loc, usecols=usecols, sep=',')
test_data = test_data.values

# Build TensorDataset and DataLoaders for the training and testing split
tr_dataset = TensorDataset(data_tensor=t.from_numpy(train_data[:, :-1]).type(t.FloatTensor),
                           target_tensor=t.from_numpy(train_data[:, -1].reshape(-1, 1)).type(t.FloatTensor))
te_dataset = TensorDataset(data_tensor=t.from_numpy(test_data[:, :-1]).type(t.FloatTensor),
                           target_tensor=t.from_numpy(test_data[:, -1].reshape(-1, 1)).type(t.FloatTensor))
del train_data
del test_data

tr_loader = DataLoader(tr_dataset, batch_size=p.batch_size, shuffle=True)
te_loader = DataLoader(te_dataset, batch_size=1000, shuffle=True)

# Build your network
main_network = nn.Sequential()
layer_list = []
with open(p.layer_info, 'r') as f:
    for line in f:
        layer_list = line.split(',')

for i in range(0, len(layer_list) - 1):
    main_network.add_module('Linear-{0}-{1}-{2}'.format(i, layer_list[i], layer_list[i + 1]),
                            nn.Linear(int(layer_list[i]), int(layer_list[i + 1])))

    if i != len(layer_list) - 2:
        if p.activation == 'relu':
            main_network.add_module('ReLU-{0}'.format(i), nn.ReLU(inplace=True))
        elif p.activation == 'sigmoid':
            main_network.add_module('Sigmoid-{0}'.format(i), nn.Sigmoid())
        elif p.activation == 'tanh':
            main_network.add_module('Tanh-{0}'.format(i), nn.Tanh())

    else:
        main_network.add_module('Sigmoid-{0}'.format(i), nn.Sigmoid())

# Build losses and optimizers
if p.loss == 'bce':
    loss = nn.BCELoss()
elif p.loss == 'mse':
    loss = nn.MSELoss()

if CUDA_CHECK:
    main_network = main_network.cuda()
    loss = loss.cuda()

optim_yaml = yaml.load(open(p.opt_det, 'r'))
if optim_yaml['name'] == 'sgd':    
    optimizer = t.optim.SGD(main_network.parameters(), lr=optim_yaml['params']['lr'])
elif optim_yaml['name'] == 'adam':
    optimizer = t.optim.Adam(main_network.parameters(), lr=optim_yaml['params']['lr'])
elif optim_yaml['name'] == 'adagrad':
    optimizer = t.optim.Adagrad(main_network.parameters(), lr=optim_yaml['params']['lr'])
elif optim_yaml['name'] == 'rmsprop':
    optimizer = t.optim.RMSprop(main_network.parameters(), lr=optim_yaml['params']['lr'])

for k in optim_yaml['params'].keys():
    optimizer.param_groups[0][k] = optim_yaml['params'][k]

# Build the training loop
n_iters = 0
train_losses = []
train_accus = []
test_losses = []
test_accus = []
for e in range(0, p.epochs):

    # Training phase
    main_network.train()
    for i, itr in enumerate(tr_loader):
        main_network.zero_grad()
        x, y = itr
        if CUDA_CHECK:
            x = x.cuda()
            y = y.cuda()
        outputs = main_network(x)
        cur_loss = loss(outputs, y)
        if n_iters % 100 == 0:
            print(round(cur_loss.data[0], 5))
        cur_loss.backward()
        optimizer.step()
        n_iters += 1

    # Evaluation phase
    main_network.eval()
    train_loss = 0.0
    train_accu = 0.0
    for i, itr in enumerate(tr_loader):
        x, y = itr
        if CUDA_CHECK:
            x = x.cuda()
            y = y.cuda()
        outputs = main_network(x)
        cur_loss = loss(outputs, y)
        outputs[outputs > 0.5] = 1
        outputs[outputs <= 0.5] = 0
        train_loss += cur_loss.data[0] * len(y)
        train_accu += (len(y) - (outputs - y).abs().sum().data[0])

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
        outputs = main_network(x)
        cur_loss = loss(outputs, y)
        outputs[outputs > 0.5] = 1
        outputs[outputs <= 0.5] = 0
        test_loss += cur_loss.data[0] * len(y)
        test_accu += (len(y) - (outputs - y).abs().sum().data[0])

    test_loss /= len(te_dataset)
    test_accu /= len(te_dataset)
    test_losses.append(test_loss)
    test_accus.append(test_accu)

plt.subplot(121)
plt.plot(list(range(1, p.epochs + 1)), train_losses, label='Train Loss')
plt.subplot(122)
plt.plot(list(range(1, p.epochs + 1)), train_accus, label='Train Accuracy')
plt.show()

plt.subplot(121)
plt.plot(list(range(1, p.epochs + 1)), test_losses, label='Test Loss')
plt.subplot(122)
plt.plot(list(range(1, p.epochs + 1)), test_accus, label='Test Accuracy')
plt.show()
