import torch
import skorch

from distributions import randomvals
from modules import FCNet, RandomFCNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def _get_optimizer_obj(opt_dict):
    """
    Function to obtain the optimizer class
    """
    if opt_dict['name'] == 'adam':
        opt_obj = torch.optim.Adam
    elif opt_dict['name'] == 'sgd':
        opt_obj = torch.optim.SGD
    elif opt_dict['name'] == 'adagrad':
        opt_obj = torch.optim.Adagrad
    elif opt_dict['name'] == 'rmsprop':
        opt_obj = torch.optim.RMSprop

    return opt_obj


def _get_criterion_obj(criterion_str):
    """
    Function to obtain the criterion class
    """
    if criterion_str == 'cross_entropy':
        criterion_obj = torch.nn.CrossEntropyLoss
    elif criterion_str == 'mse':
        criterion_obj = torch.nn.MSELoss
    elif criterion_str == 'bce':
        criterion_obj = torch.nn.BCELoss

    return criterion_obj


class NNetClassifier(skorch.NeuralNetClassifier):
    """
    Derived class from skorch.NeuralNetClassifier
    This class instantiates a Classifier based on a torch nn.Module, and certain other parameters

    Args:
        fcnet       : FCNet object obtained from modules
        criterion   : Loss criterion object
        optimizer   : dictionary stating name of optimizer and learning rate. Format is:
                        optimizer = {'name': <name>, 'lr': <learning rate>}
        max_epochs  : number of epochs to train for
        batch_size  : batch size to take for training
        use_cuda    : boolean to use CUDA
    """
    def __init__(self, fcnet, criterion, optimizer, max_epochs, batch_size, use_cuda=False):
        assert isinstance(fcnet, FCNet), "fcnet must be an instance of FCNet"

        criterion_obj = _get_criterion_obj(criterion)
        optimizer_obj = _get_optimizer_obj(optimizer)
        super(NNetClassifier, self).__init__(module=fcnet, criterion=criterion_obj, optimizer=optimizer_obj,
                                             lr=optimizer['lr'], max_epochs=max_epochs, batch_size=batch_size,
                                             use_cuda=use_cuda)


class RandomNNetClassifier(skorch.NeuralNetClassifier):
    """
    Dervied class from skorch.NeuralNetClassifier
    This class instantiates a random network architecture based on RandomFCNet.

    Args:
        input_dim   : Input dimensions
        output_dim  : Output dimensions
        end_activation      : Activation to use in the output layer
    Optional args:
        nlayers     : Number of hidden layers in the network
        criterion   : Loss criterion object (default: None (meaning CrossEntropyLoss))
        optimizer   : dictionary stating name of optimizer and learning rate. Format is:
                        optimizer = {'name': <name>, 'lr': <learning rate>}
                      (default: None (meaning Adam with lr = 3e-04))
        max_epochs  : number of epochs to train for (default: 10)
        batch_size  : batch size to take for training (default: 64)
        use_cuda    : boolean to use CUDA (default: False)
    """
    def __init__(self, input_dim, output_dim, end_activation, nlayers=None, criterion=None,
                 optimizer=None, max_epochs=10, batch_size=64, use_cuda=False, *args, **kwargs):
        model = RandomFCNet(input_dim=input_dim, output_dim=output_dim, end_activation=end_activation, nlayers=nlayers)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.end_activation = end_activation

        # Get criterion
        if criterion is not None:
            criterion_obj = _get_criterion_obj(criterion)
        else:
            criterion_obj = torch.nn.CrossEntropyLoss

        # Get optimizer
        if optimizer is not None:
            optimizer_obj = _get_optimizer_obj(optimizer)
            learning_rate = optimizer['lr']
        else:
            optimizer_obj = torch.optim.Adam
            learning_rate = 3e-04

        super(RandomNNetClassifier, self).__init__(module=model, criterion=criterion_obj, optimizer=optimizer_obj,
                                                   lr=learning_rate, max_epochs=max_epochs, batch_size=batch_size,
                                                   use_cuda=use_cuda, *args, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        super(RandomNNetClassifier, self).fit(X, y, *args, **kwargs)


def get_generic_module(fcnet, criterion, optimizer, epochs, batch_size, use_cuda=False):
    """
    Constructor function for NNetClassifier
    """
    return NNetClassifier(fcnet=fcnet, criterion=criterion, optimizer=optimizer,
                          max_epochs=epochs, batch_size=batch_size, use_cuda=use_cuda)


def grid_search_module(fcnet, criterion, optimizer, epochs,
                       batch_size, params, use_cuda=False,
                       n_jobs=2):
    """
    Function to return an instance of Grid Search module
    Args:
        fcnet       : FCNet object obtained from modules
        criterion   : Loss criterion object
        optimizer   : dictionary stating name of optimizer and learning rate. Format is:
                        optimizer = {'name': <name>, 'lr': <learning rate>}
        epochs      : number of epochs to train for
        batch_size  : batch size to take for training
        params      : dict of parameters and their respective values. Format is:
                        {'param1': [<list of vals for param1>], 'param2': [<list of vals for param2], ...}
                      Name of parameters should follow that specified in skorch docs.
        use_cuda    : boolean to use CUDA
    """
    for k in params.keys():
        assert isinstance(params[k], list), "The possible values should be passed as a list"
    nnclassifier = get_generic_module(fcnet=fcnet, criterion=criterion, optimizer=optimizer,
                                      epochs=epochs, batch_size=batch_size, use_cuda=use_cuda)
    gs = GridSearchCV(nnclassifier, params, refit=False, scoring='neg_log_loss', n_jobs=n_jobs, verbose=0, cv=2)
    return gs


def random_search_module(fcnet, criterion, optimizer, epochs,
                         batch_size, param_dist, use_cuda=False,
                         n_iter=25, n_jobs=2):
    """
    Function to return an instance of Grid Search module
    Args:
        fcnet       : FCNet object obtained from modules
        criterion   : Loss criterion object
        optimizer   : dictionary stating name of optimizer and learning rate. Format is:
                        optimizer = {'name': <name>, 'lr': <learning rate>}
        epochs      : number of epochs to train for
        batch_size  : batch size to take for training
        param_dist  : dict of parameters and their respective distributions. Format is:
                        {'param1': distribution1, 'param2': distribution2, ...}
                      Name of parameters should follow that specified in skorch docs.
        use_cuda    : boolean to use CUDA
    """
    for k in param_dist.keys():
        assert issubclass(param_dist[k].__class__, randomvals) or isinstance(param_dist[k], list),\
            "The probability distributions should be one of randombool, randomtext or randomintervals or a list"
    nnclassifier = get_generic_module(fcnet=fcnet, criterion=criterion, optimizer=optimizer,
                                      epochs=epochs, batch_size=batch_size, use_cuda=use_cuda)
    rs = RandomizedSearchCV(nnclassifier, param_dist, n_iter=n_iter, refit=False,
                            scoring='neg_log_loss', n_jobs=n_jobs, verbose=0, cv=2)
    return rs
