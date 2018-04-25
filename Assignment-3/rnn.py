import numpy as np


def softmax(x, T=1):
    """
    Computes softmax for a vector x, with temperature T
    """
    tmp = np.exp(x) / T
    return tmp / (tmp.sum() / T)


class myRNN(object):
    """
    Class to represent an RNN object.

    Args:
        vocab_size  : Size of the vocabulary
    Optional Args:
        hidden_size : Size of the hidden layer
    """
    def __init__(self, vocab_size, hidden_size=250, seq_length=5, temperature=1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.temperature = temperature

        # Initialize weights
        fan_in, fan_out = vocab_size, hidden_size
        factor = np.sqrt(2 / (fan_in + fan_out))
        self.Wxh = np.random.uniform(-factor, factor)
        self.Whh = np.random.uniform(-factor, factor)
        self.Who = np.random.uniform(-factor, factor)

    def forward(self, inputs, previous_hidden_state):
        """
        Function to take in one-hot-encoded inputs, and the previous hidden state

        Args:
            inputs      : Sequence of one hot encoded inputs (input_size, vocab_size)
            previous_hidden_state   : Previous hidden state
        Returns:
            hidden_states, prob_outputs : hidden states computed and character probabilities
        """
        hidden_states = {}
        prob_outputs = {}
        hidden_states[-1] = np.copy(previous_hidden_state.reshape(-1, 1))
        for t in range(0, len(inputs)):
            inputs[t] = inputs[t].reshape(-1, 1)
            hidden_states[t] = np.tanh(np.dot(self.Wxh, inputs[t]) + np.dot(self.Whh, hidden_states[t - 1]))
            raw_outputs = np.dot(self.Who, hidden_states[t])
            prob_outputs[t] = softmax(raw_outputs, self.temperature).reshape(-1)
        return hidden_states, prob_outputs, loss

    def loss(self, pred_probs, targets):
        """
        Function to compute the cross entropy loss

        Args:
            pred_probs  : predicted probabilities for characters
            targets     : actual targets
        """
        loss = 0.0
        for t in range(0, targets):
            loss += -np.log(pred_probs[t][targets[t]])
        return loss

    def backward(self):
        raise NotImplementedError
