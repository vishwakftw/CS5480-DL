import text
import numpy as np

from rnn import myRNN
from matplotlib import pyplot as plt
from argparse import ArgumentParser as AP

REG_FACT = 1e-04
VERBOSE = False
VOCAB_SIZE = 128

p = AP()
p.add_argument('--seq_length', type=int, default=25, help='Length of unrolled RNN sequence')
p.add_argument('--temperature', type=float, default=1.0, help='Temperature to consider for softmax')
p.add_argument('--lr', type=float, default=1e-03, help='Learning rate')
p.add_argument('--epochs', type=int, default=100, help='Number of epochs')
p.add_argument('--text_src', type=str, required=True, help='Text source to read from')
p.add_argument('--hidden', type=int, default=250, help='Number of dimensions in the hidden layer')
p.add_argument('--hidden_init', type=str, default='zeros', choices=['random', 'zeros'], help='Initialize the hidden state')
p.add_argument('--checkpoint_interval', type=int, default=20, help='Checkpoint (i.e., generate) every certain epochs')
p.add_argument('--choice', type=str, default='hard', help='Selection of character (soft / hard)')
p.add_argument('--graph', action='store_true', help='Toggle to save a graph at the end of training')
p.add_argument('--nsample', type=int, default=100, help='Number of characters')
p = p.parse_args()

RNN = myRNN(vocab_size=VOCAB_SIZE, hidden_size=p.hidden, temperature=p.temperature)

# get the text
all_text = open(p.text_src, 'r', encoding='ascii', errors='ignore').read()
text_len = len(all_text)
print("Length of text: {}".format(text_len))
encoded_text = text.get_char_to_encoding(all_text)

# initialize some variables
epoch = 0
iters = 0
pointer = 0
if p.hidden_init == 'zeros':
    hidden_state = np.zeros(p.hidden).reshape(-1, 1)
if p.hidden_init == 'random':
    hidden_state = np.random.randn(p.hidden).reshape(-1, 1)

losses = []
while True:
    try:
        if pointer + p.seq_length + 1 >= text_len:    # we don't have enough inputs, we take it till the end
            inputs = encoded_text[pointer : -1]
            targets = encoded_text[pointer + 1 : ]
            # reset pointer
            pointer = 0

            # reset memory
            if p.hidden_init == 'zeros':
                hidden_state = np.zeros(p.hidden).reshape(-1, 1)
            if p.hidden_init == 'random':
                hidden_state = np.random.randn(p.hidden).reshape(-1, 1)

            # increment epochs
            epoch += 1

        else:
            inputs = encoded_text[pointer : pointer + p.seq_length]
            targets = encoded_text[pointer + 1 : pointer + p.seq_length + 1]

        # Training
        loss, grads, hidden_state = RNN.backward(inputs=inputs, targets=targets, previous_hidden_state=hidden_state)
        if p.graph and pointer == 0:
            losses.append(loss)
        iters += 1

        if VERBOSE:
            if iters % 1000 == 1:
                print('loss at iteration {} = {}'.format(iters, loss))
        # Gradient step
        RNN.Wxh -= p.lr * (grads[0] + REG_FACT * RNN.Wxh)
        RNN.Whh -= p.lr * (grads[1] + REG_FACT * RNN.Whh)
        RNN.Who -= p.lr * (grads[2] + REG_FACT * RNN.Who)

        if epoch % p.checkpoint_interval == 0 and pointer == 0 and epoch != 0:
            print("##### Checkpoint start : Epoch {} #####".format(epoch))
            gen_text = RNN.get(seed=encoded_text[0], hidden_state=hidden_state, n=p.nsample, choice=p.choice)
            act_text = text.get_encoding_to_char(gen_text)
            print("".join(act_text))
            print("##### Checkpoint end : Epoch {} #####".format(epoch))

        if epoch == p.epochs:
            break

        pointer += p.seq_length

    except KeyboardInterrupt:
        break

if p.graph:
    plt.figure(figsize=(12, 10))
    plt.title("Variation of loss with epochs", fontsize=15)
    plt.plot(range(1, len(losses) + 1), losses, 'b', linewidth=3.0)
    plt.savefig('RNN_training.png', dpi=100)
