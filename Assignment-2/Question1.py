import torch as t
import pandas as pd
import torch.nn as nn
from argparse import ArgumentParser as AP
from torch.utils.data import TensorDataset, DataLoader

p = AP()
p.add_argument('--train_data_loc', type=str, required=True, help='Root location for the training data')
p.add_argument('--test_data_loc', type=str, required=True, help='Root location for the testing data')
p.add_argument('--batch_size', type=int, default=100, help='Batch size for the training')
p.add_argument('--opt_det', type=str, default=1e-04, help='Root location for the Optimizer details stored as YAML')
p.add_argument('--layer_info', type=str, required=True, help='Root location for CSV file with MLP layer information')
p.add_argument('--activation', type=str, default='relu', help='Activation for hidden layers')
p = p.parse_args()

# Read the dataset from the file
usecols = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Occupancy']
train_data = pd.read_csv(p.train_data_loc, usecols=usecols, sep=',')
train_data = train_data.values

test_data = pd.read_csv(p.test_data_loc, usecols=usecols, sep=',')
test_data = test_data.values

# Build TensorDataset and DataLoaders for the training and testing split
tr_dataset = TensorDataset(data_tensor=t.from_numpy(train_data[:, :-1]), target_tensor=t.from_numpy(train_data[:, -1]))
te_dataset = TensorDataset(data_tensor=t.from_numpy(test_data[:, :-1]), target_tensor=t.from_numpy(test_data[:, -1]))
del train_data
del test_data

tr_loader = DataLoader(tr_dataset, batch_size=p.batch_size, shuffle=True)
te_loader = DataLoader(te_dataset, batch_size=3251, shuffle=True)

# Build your network
main_network = nn.Sequential()
layer_list = []
with open(p.layer_info, 'r') as f:
    for line in f:
        layer_list = line.split(',')

for i in range(0, len(layer_list) - 1):
    main_network.add_module('Linear-{0}-{1}-{2}'.format(i, layer_list[i], layer_list[i + 1]),
                            nn.Linear(layer_list[i], layer_list[i + 1]))

    if i != len(layer_list) - 2:
        if p.activation == 'relu':
            main_network.add_module('ReLU-{0}'.format(i), nn.ReLU(inplace=True))
        elif p.activation == 'sigmoid':
            main_network.add_module('Sigmoid-{0}'.format(i), nn.Sigmoid())
        elif p.activation == 'tanh':
            main_network.add_module('Tanh-{0}'.format(i), nn.Tanh())

    else:
        main_network.add_module('Sigmoid-{0}'.format(i), nn.Sigmoid())
