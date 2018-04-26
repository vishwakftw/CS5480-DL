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
        hidden_size : Size of the hidden layer (default=250)
        temperature : Temperature for probabilities by softmax.
                      Lower temperature indicates softer, higher indicates harder (default=1)
    """
    def __init__(self, vocab_size, hidden_size=250, temperature=1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.temperature = temperature

        # Initialize weights
        fan_in, fan_out = vocab_size, hidden_size
        factor = np.sqrt(2 / (fan_in + fan_out))
        self.Wxh = np.random.uniform(-factor, factor, size=(self.hidden_size, self.vocab_size))
        self.Whh = np.random.uniform(-factor, factor, size=(self.hidden_size, self.hidden_size))
        self.Who = np.random.uniform(-factor, factor, size=(self.vocab_size, self.hidden_size))

    def forward(self, inputs, previous_hidden_state):
        """
        Function to take in one-hot-encoded inputs, and the previous hidden state

        Args:
            inputs      : Sequence of characters (integers ASCII ordinal)
            previous_hidden_state   : Previous hidden state
        Returns:
            hidden_states, prob_outputs : hidden states computed and character probabilities
        """
        hidden_states = {}
        prob_outputs = {}
        hidden_states[-1] = np.copy(previous_hidden_state)
        for t in range(0, len(inputs)):
            inputs_ = np.zeros(self.vocab_size)
            inputs_[inputs[t]] = 1
            inputs_ = inputs_.reshape(-1, 1)
            hidden_states[t] = np.tanh(np.matmul(self.Wxh, inputs_) + np.matmul(self.Whh, hidden_states[t - 1]))
            raw_outputs = np.matmul(self.Who, hidden_states[t])
            prob_outputs[t] = softmax(raw_outputs, self.temperature)
        return hidden_states, prob_outputs

    def loss(self, pred_probs, targets):
        """
        Function to compute the cross entropy loss

        Args:
            pred_probs  : predicted probabilities for characters
            targets     : sequence of actual targets (integers ASCII ordinal)
        """
        loss = 0.0
        for t in range(0, len(targets)):
            loss += -np.log(pred_probs[t][targets[t]])
        return loss

    def backward(self, inputs, targets, previous_hidden_state):
        """
        Function to perform one BPTT

        Args:
            inputs      : Sequence of characters (integers ASCII ordinal)
            targets     : Sequence of actual targets (integers ASCII ordinal)
            previous_hidden_state   : Previous hidden state
        Returns:
            loss, gradients, last_hidden_state  : Loss value, gradients (tuple) and last hidden state
        """
        hidden_states, char_probs = self.forward(inputs, previous_hidden_state)
        cur_loss = self.loss(char_probs, targets)
        grad_Wxh, grad_Whh, grad_Who = np.zeros(self.Wxh.shape), np.zeros(self.Whh.shape), np.zeros(self.Who.shape)
        do_dh_next_timestep = np.zeros(hidden_states[0].shape)
        for t in range(len(inputs) - 1, 0, -1):
            delta_output = np.copy(char_probs[t])

            # First step: backprop to the output layer
            delta_output[targets[t]] -= 1
            # Second step: backprop to the weights of the output layer
            grad_Who += np.matmul(delta_output, hidden_states[t].T)

            # Now come to the tanh non-linearity
            do_d_tanh = (1 - hidden_states[t] * hidden_states[t])
            do_dh = do_d_tanh * (np.matmul(self.Who.T, delta_output) + do_dh_next_timestep)

            # Regenerate the inputs
            inputs_ = np.zeros(self.vocab_size)
            inputs_[inputs[t]] = 1
            inputs_ = inputs_.reshape(-1, 1)
            grad_Wxh += np.matmul(do_dh, inputs_.T)
            grad_Whh += np.matmul(do_dh, hidden_states[t - 1].T)
            do_dh_next_timestep = np.matmul(self.Whh.T, do_dh)

        # Clipping the gradients to avoid gradient explode
        grad_Wxh = grad_Wxh.clip(min=-1, max=1)
        grad_Whh = grad_Whh.clip(min=-1, max=1)
        grad_Who = grad_Who.clip(min=-1, max=1)

        return cur_loss, (grad_Wxh, grad_Whh, grad_Who), hidden_states[len(inputs) - 1]

    def get(self, seed, hidden_state, n, choice='soft'):
        """
        Function to get some characters from the model

        Args:
            seed    : Start seed for the prediction
            hidden_state    : Initial hidden state to get things moving
            n   : number of characters to sample
        Optional Args:
            choice  : 'soft' for using a probability distribution, 'hard' for using maximum probability
        Returns:
            Sequence of integers (ASCII ordinals)
        """
        inp = np.zeros(self.vocab_size)
        inp[seed] = 1
        inp = inp.reshape(-1, 1)
        char_vals = []
        for t in range(0, n):
            hidden_state = np.tanh(np.matmul(self.Wxh, inp) + np.matmul(self.Whh, hidden_state))
            otp = np.matmul(self.Who, hidden_state)
            otp = softmax(otp)
            # We take the characters in a soft manner, not a hard manner
            if choice == 'soft':
                char = np.random.choice(list(range(self.vocab_size)), p=otp.ravel())
            elif choice == 'hard':
                char = np.argmax(otp.reshape(-1))

            char_vals.append(char)
            # make the next input
            inp = np.zeros(self.vocab_size)
            inp[char] = 1
            inp = inp.reshape(-1, 1)
        return char_vals
