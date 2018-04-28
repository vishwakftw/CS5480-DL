import numpy as np


def softmax(x, T=1):
    """
    Computes softmax for a vector x, with temperature T
    """
    tmp = np.exp(x) / T
    return tmp / (tmp.sum() / T)


class myLSTM(object):
    """
    Class to represent an LSTM object

    Args:
        vocab_size  : Size of the vocabulary
    Optional Args:
        hidden_size : Size of the hidden layer (default=250)
    """
    def __init__(self, vocab_size, hidden_size=250):
        raise NotImplementedError("LSTM not yet ready")
