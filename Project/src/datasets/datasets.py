import torch
import numpy as np
from torchfile import load


def get_MNIST_dataset(train_root_location, test_root_location):
    """
    Function to obtain a tensor-ized version for the MNIST dataset
    Args:
        train_root_location : string with location
        test_root_location  : string with location
    """
    pure_tr_dataset = torch.load(train_root_location)
    pure_te_dataset = torch.load(test_root_location)
    tr_input = (pure_tr_dataset[0].type(torch.FloatTensor) / 255).view(-1, 784)
    te_input = (pure_te_dataset[0].type(torch.FloatTensor) / 255).view(-1, 784)
    tr_input.sub_(0.1307).div_(0.3081)
    te_input.sub_(0.1307).div_(0.3081)
    return (tr_input.numpy(), pure_tr_dataset[1].numpy()), (te_input.numpy(), pure_te_dataset[1].numpy())


def get_CIFAR10_dataset(train_root_location, test_root_location):
    """
    Function to obtain a tensor-ized version for the CIFAR10 dataset
    Args:
        train_root_location : dictionary of the form for the training data
                                    {'inputs': <input_root>, 'outputs': <output_root>}
        test_root_location  : dictionary of the form for the testing data
                                    {'inputs': <input_root>, 'outputs': <output_root>}
    """
    tr_input = load(train_root_location['inputs']).astype(np.float32)
    tr_output = load(train_root_location['outputs']).astype(int)
    te_input = load(test_root_location['inputs']).astype(np.float32)
    te_output = load(test_root_location['outputs']).astype(int)
    return (tr_input, tr_output - 1), (te_input, te_output - 1)


def get_SVHN_dataset(train_root_location, test_root_location):
    """
    Function to obtain a tensor-ized version for the SVHN dataset
    Args:
        train_root_location : dictionary of the form for the training data
                                    {'inputs': <input_root>, 'outputs': <output_root>}
        test_root_location  : dictionary of the form for the testing data
                                    {'inputs': <input_root>, 'outputs': <output_root>}
    """
    tr_input = torch.load(train_root_location['inputs'])
    tr_input.sub_(0.5).div_(0.5)
    tr_output = torch.load(train_root_location['outputs']).numpy()
    te_input = torch.load(test_root_location['inputs'])
    te_input.sub_(0.5).div_(0.5)
    te_output = torch.load(test_root_location['outputs']).numpy()
    return (tr_input.numpy(), tr_output), (te_input.numpy(), te_output)
