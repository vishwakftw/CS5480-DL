from functools import partial

import numpy as np
import torch.nn as nn


def _get_activation(activation_str):
    """
    Function to get activation function instance
    """
    if activation_str == 'relu':
        return nn.ReLU(inplace=True)
    if activation_str == 'sigmoid':
        return nn.Sigmoid()
    if activation_str == 'tanh':
        return nn.Tanh()
    if activation_str == 'softmax':
        return nn.Softmax(dim=1)
    return None


def _init_scheme(mdl, init_str):
    """
    Function to perform weight initialization based on scheme
    """
    if init_str == 'xavier':
        initializer = nn.init.xavier_uniform
        if isinstance(mdl, nn.Linear):
            initializer(mdl.weight.data)
            mdl.bias.data.fill_(0)

    if init_str == 'kaiming':
        initializer = nn.init.kaiming_uniform
        if isinstance(mdl, nn.Linear):
            initializer(mdl.weight.data)
            mdl.bias.data.fill_(0)

    if init_str == 'uniform':
        initializer = nn.init.uniform
        if isinstance(mdl, nn.Linear):
            initializer(mdl.weight.data)


class FCNet(nn.Module):
    """
    This is a module to construct a given neural network architecture based on specifications
    Characteristics taken care of:
    1. Existence of Dropout in every layer, and how much probable is dropout
    2. Existence of Batch Normalization in every layer
    3. 4 shapes - decreasing, convex (decreasing - increasing), concave (increasing - decreasing), constant
    4. Factor of decrease / increase
    5. Number of layers
    6. Initialization schemes
    7. Activation function

    Args:
        input_dim   : Input dimensions
        hidden_structure    : One of 'decreasing', 'convex', 'concave', 'constant'
        hidden_factor1, hidden_factor2  : Shape control parameters
        nlayers     : Number of hidden layers
        output_dim  : Output dimensions
        dropout_probs       : Vector with dropout probabilities
        bnorms              : Boolean vector with boolean whether to include batch normalization or not
        init_scheme         : One of 'xavier', 'kaiming', 'uniform', 'standard'
        activation          : Activation to use in between layers
        end_activation      : Activation to use in the output layer
    """
    def __init__(self, input_dim, hidden_structure, hidden_factor1, hidden_factor2,
                 output_dim, nlayers, dropout_probs, bnorms, init_scheme,
                 activation, end_activation):
        super(FCNet, self).__init__()

        assert len(bnorms) == nlayers, "Batch Normalization configuration does not satisfy length"
        assert len(dropout_probs) == nlayers, "Dropout configuration does not satisfy length"
        assert hidden_structure in ['decreasing', 'convex', 'concave', 'constant'],\
            "Hidden structure specified doesn't match existing options"

        self.networks = nn.Sequential()

        # Construct the structure of the network
        arch = [input_dim]
        if hidden_structure == 'constant':
            hdim = int(hidden_factor1 * input_dim)
            arch += ([hdim] * (nlayers // 2))
            if hidden_factor2 is None:
                hdim = int(hidden_factor1 * hdim)
                arch += ([hdim] * (nlayers - nlayers // 2))
            else:
                hdim = int(hidden_factor2 * hdim)
                arch += ([hdim] * (nlayers - nlayers // 2))

        if hidden_structure == 'decreasing':
            assert hidden_factor1 < 1 and hidden_factor2 < 1, "Incorrect shape parameters"
            hdim = input_dim
            for _ in range(0, nlayers // 2):
                hdim = int(hidden_factor1 * hdim)
                arch.append(hdim)

            for _ in range(nlayers // 2, nlayers):
                hdim = int(hidden_factor2 * hdim)
                arch.append(hdim)

        if hidden_structure == 'convex':
            assert hidden_factor1 < 1 and hidden_factor2 > 1, "Incorrect shape parameters"
            hdim = input_dim
            for _ in range(0, nlayers // 2):
                hdim = int(hidden_factor1 * hdim)
                arch.append(hdim)

            for _ in range(nlayers // 2, nlayers):
                hdim = int(hidden_factor2 * hdim)
                arch.append(hdim)

        if hidden_structure == 'concave':
            assert hidden_factor1 > 1 and hidden_factor2 < 1, "Incorrect shape parameters"
            hdim = input_dim
            for _ in range(0, nlayers // 2):
                hdim = int(hidden_factor1 * hdim)
                arch.append(hdim)

            for _ in range(nlayers // 2, nlayers):
                hdim = int(hidden_factor2 * hdim)
                arch.append(hdim)

        arch.append(output_dim)

        # Construct full network
        act_fn = _get_activation(activation)
        for i in range(0, len(arch) - 1):
            self.networks.add_module('FC-{0}-{1}-{2}'.format(i, arch[i], arch[i + 1]),
                                     nn.Linear(arch[i], arch[i + 1]))
            if i == len(arch) - 2:
                continue

            if bnorms[i]:
                self.networks.add_module('BatchNorm-{0}-{1}'.format(i, arch[i + 1]),
                                         nn.BatchNorm1d(arch[i + 1]))

            if act_fn is not None:
                self.networks.add_module('{0}-{1}'.format(act_fn.__class__.__name__, i), act_fn)

            if dropout_probs[i] != 0:
                self.networks.add_module('Dropout-{0}-{1}'.format(i, dropout_probs[i]),
                                         nn.Dropout(p=dropout_probs[i]))

        end_act_fn = _get_activation(end_activation)
        if end_act_fn is not None:
            self.networks.add_module('{0}-{1}'.format(end_act_fn.__class__.__name__, i), end_act_fn)

        # Apply initialization scheme
        self.init_scheme = init_scheme
        self.net_init()

    def forward(self, input):
        """
        Performs the forward pass on a set of input data
        """
        return self.networks(input)


    def net_init(self):
        """
        Applies initialization scheme
        """
        self.networks.apply(partial(_init_scheme, init_str=self.init_scheme))


class RandomFCNet(FCNet):
    """
    This is a derived class of the FCNet module, where one can specify the dimensionality of
    inputs, outputs and the activation at the last layer and the class will instantiate a randomized architecture

    The number of layers in the randomized architecture can go up to 5, but can be exceeded if the number is
    explicitly specified (optional, however)

    Args:
        input_dim   : Input dimensions
        output_dim  : Output dimensions
        end_activation      : Activation to use in the output layer
    Optional args:
        nlayers     : Number of hidden layers in the network
    """
    def __init__(self, input_dim, output_dim, end_activation, nlayers=None):

        # Max layers
        if nlayers is None:
            nlayers = np.random.randint(low=1, high=5)

        self.nlayers = nlayers

        # Get network's structural parameters
        hidden_structure = np.random.choice(['decreasing', 'convex', 'concave', 'constant'])

        if hidden_structure == 'decreasing':
            hidden_factor1 = round(np.random.choice(np.arange(0.5, 0.975, 0.025)), 3)
            hidden_factor2 = round(np.random.choice(np.arange(0.5, 0.975, 0.025)), 3)

        elif hidden_structure == 'convex':
            hidden_factor1 = round(np.random.choice(np.arange(0.5, 0.975, 0.025)), 3)
            hidden_factor2 = round(np.random.choice(np.arange(1.025, 1.525, 0.025)), 3)

        elif hidden_structure == 'concave':
            hidden_factor1 = round(np.random.choice(np.arange(1.025, 1.525, 0.025)), 3)
            hidden_factor2 = round(np.random.choice(np.arange(0.5, 0.975, 0.025)), 3)

        elif hidden_structure == 'constant':
            hidden_factor1 = round(np.random.choice(np.arange(0.5, 1.525, 0.025)), 3)
            hidden_factor2 = None

        # Dropout
        dropout_probs = list(np.around(np.random.choice(np.arange(0.0, 0.325, 0.025), nlayers), 3))

        # Batch Normalization
        bnorms = list(np.random.choice([True, False], nlayers))

        # Initialization schemes and activations
        init_scheme = np.random.choice(['kaiming', 'xavier', 'standard'])
        activation = np.random.choice(['relu', 'sigmoid', 'tanh'])

        super(RandomFCNet, self).__init__(input_dim=input_dim, hidden_structure=hidden_structure,
                                          hidden_factor1=hidden_factor1, hidden_factor2=hidden_factor2,
                                          output_dim=output_dim, nlayers=nlayers, dropout_probs=dropout_probs,
                                          bnorms=bnorms, init_scheme=init_scheme, activation=activation,
                                          end_activation=end_activation)
