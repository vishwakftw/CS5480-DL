import numpy as np


def softmax(x, T=1):
    """
    Computes softmax for a vector x, with temperature T
    """
    tmp = np.exp(x) / T
    return tmp / (tmp.sum() / T)


def sigmoid(x):
    """
    Computes sigmoid for a vector x
    """
    return 1 / (1 + np.exp(-x))

class myLSTM(object):
    """
    Class to represent an LSTM object

    Args:
        vocab_size  : Size of the vocabulary
    Optional Args:
        hidden_size : Size of the hidden layer (default=250)

    Equations for forward are:
        i_{t} = sigmoid(W_{xi}x_{t} + W_{hi}h_{t-1}) = sigmoid(W_{i}[x_{t} (cat) h_{t-1}])
        f_{t} = sigmoid(W_{f}[x_{t} (cat) h_{t-1}])
        mc_{t} = \hat{c}_{t} = tanh(W_{c}[x_{t} (cat) h_{t-1}])
        c_{t} = f_{t} c_{t-1} + i_{t} mc_{t}
        o_{t} = sigmoid(W_{o}[x_{t} (cat) h_{t-1}])
        h_{t} = o_{t} tanh(c_{t})
        v_{t} = \hat{y}_{t} = softmax(W_{v}h_{t})

    Idea for concatenating is borrowed from http://blog.varunajayasiri.com/numpy_lstm.html
    """
    def __init__(self, vocab_size, hidden_size=100):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Initialize weights
        d1 = hidden_size
        d2 = vocab_size
        val = np.sqrt(2 / (2*d1 + d2))
        self.Wi = np.random.uniform(-val, val, size=(d1, d1 + d2))
        self.Wf = np.random.uniform(-val, val, size=(d1, d1 + d2))
        self.Wc = np.random.uniform(-val, val, size=(d1, d1 + d2))
        self.Wo = np.random.uniform(-val, val, size=(d1, d1 + d2))
        self.Wv = np.random.uniform(-np.sqrt(2 / (d2 + d1)), np.sqrt(2 / (d2 + d1)), size=(d2, d1))

    def forward_step(self, input_, previous_hidden_state, previous_cell_state):
        """
        Function to take in one hot encoded input, the previous hidden state and previous cell state

        Args:
            input_      : One hot encoded input
            previous_hidden_state   : Previous hidden state
            previous_cell_state     : Previous cell state
        """
        xh = np.vstack([input_, previous_hidden_state])
        i = sigmoid(np.matmul(self.Wi, xh))
        f = sigmoid(np.matmul(self.Wf, xh))
        mc = np.tanh(np.matmul(self.Wc, xh))
        c = f * previous_cell_state + i * mc
        o = sigmoid(np.matmul(self.Wo, xh))
        h = o * np.tanh(c)
        v = np.matmul(self.Wv, h)
        y = softmax(v)
        return xh, i, f, mc, c, o, h, v, y

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

    def backward(self, inputs, targets, previous_hidden_state, previous_cell_state):
        """
        Function to perform one BPTT

        Args:
            inputs      : Sequence of characters (integers ASCII ordinal)
            targets     : Sequence of actual targets (integers ASCII ordinal)
            previous_hidden_state   : Previous hidden state
            previous_cell_state     : Previous cell state
        Returns:
            loss, gradients, last_hidden_state, last_cell_state:
                    Loss value, gradients (tuple), last hidden state, last cell state
        """
        # Value cache
        inputs_, xhs_, is_, fs_, mcs_, cs_, os_, hs_, vs_, ys_ = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        hs_[-1] = np.copy(previous_hidden_state)
        cs_[-1] = np.copy(previous_cell_state)

        # Perform a full forward pass over all inputs
        for t in range(0, len(inputs)):
            inputs_[t] = np.zeros(self.vocab_size).reshape(-1, 1)
            inputs_[t][inputs[t]] = 1
            xh, i, f, mc, c, o, h, v, y = self.forward_step(inputs_[t], hs_[t - 1], cs_[t - 1])
            xhs_[t], is_[t], fs_[t], mcs_[t], cs_[t], os_[t], hs_[t], vs_[t], ys_[t] = xh, i, f, mc, c, o, h, v, y

        cur_loss = self.loss(ys_, targets)
        grad_Wi, grad_Wf = np.zeros(self.Wi.shape), np.zeros(self.Wf.shape)
        grad_Wc, grad_Wo, grad_Wv = np.zeros(self.Wc.shape), np.zeros(self.Wo.shape), np.zeros(self.Wv.shape)
        dh_next_timestep = np.zeros(hs_[0].shape)
        dc_next_timestep = np.zeros(cs_[0].shape)

        for t in range(len(inputs) - 1, 0, -1):
            delta_output = np.copy(ys_[t])

            # First step: backprop to the output layer
            delta_output[targets[t]] -= 1
            grad_Wv += np.matmul(delta_output, hs_[t].T)

            # Second step: backprop to hidden state and the output connected to it
            dh = np.matmul(self.Wv.T, delta_output)
            dh += dh_next_timestep
            do = dh * np.tanh(cs_[t]) * os_[t] * (1 - os_[t])
            grad_Wo += np.matmul(do, xhs_[t].T)

            # Third step: backprop to cell state
            dc = dh * os_[t] * (1 - np.tanh(cs_[t]) * np.tanh(cs_[t]))
            dc += dc_next_timestep
            dmc = dc * is_[t] * (1 - mcs_[t] * mcs_[t])
            grad_Wc += np.matmul(dmc, xhs_[t].T)

            # Fourth step: backprop to forward gate
            df = dc * cs_[t - 1] * fs_[t] * (1 - fs_[t])
            grad_Wf += np.matmul(df, xhs_[t].T)

            # Fifth step: backprop to input gate
            di = dc * mcs_[t] * (1 - is_[t]) * is_[t]
            grad_Wi += np.matmul(di, xhs_[t].T)

            # Due to the concatenation, we need only a slice of the accumulated gradients for dh_next_timestep
            dh_next_timestep = (np.matmul(self.Wf.T, df) +\
                                np.matmul(self.Wi.T, di) +\
                                np.matmul(self.Wc.T, dmc) +\
                                np.matmul(self.Wo.T, do))[self.vocab_size:, :]
            dc_next_timestep = fs_[t] * dc

        # Clipping the gradients to avoid gradient explosion
        grad_Wi = grad_Wi.clip(min=-2, max=2)
        grad_Wf = grad_Wf.clip(min=-2, max=2)
        grad_Wc = grad_Wc.clip(min=-2, max=2)
        grad_Wo = grad_Wo.clip(min=-2, max=2)
        grad_Wv = grad_Wv.clip(min=-2, max=2)

        return cur_loss, (grad_Wi, grad_Wf, grad_Wc, grad_Wo, grad_Wv), hs_[len(inputs) - 1], cs_[len(inputs) - 1]

    def get(self, seed, hidden_state, cell_state, n):
        """
        Function to get some characters from the model

        Args:
            seed    : Start seed for the prediction
            hidden_state    : Initial hidden state to get things moving
            cell_state  : Initial cell state to get things moving
            n   : number of characters to sample
        Returns:
            Sequence of integers (ASCII ordinals)
        """
        inp = np.zeros(self.vocab_size)
        inp[seed] = 1
        inp = inp.reshape(-1, 1)
        char_vals = []
        for t in range(0, n):
            _, _, _, _, cell_state, _, hidden_state, _, otp = self.forward_step(inp, hidden_state, cell_state)
            char = np.argmax(otp.reshape(-1))
            char_vals.append(char)
            # make the next input
            inp = np.zeros(self.vocab_size)
            inp[char] = 1
            inp = inp.reshape(-1, 1)
        return char_vals
