import yaml
import modules as mdl
import datasets as dset
import skorchModules as skm

from itertools import product
from argparse import ArgumentParser as AP
from numpy import arange, around

p = AP()
p.add_argument('--params_src', type=str, required=True, help='Source YAML file for the params')
p.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10', 'svhn'],
               help='Dataset name')
p.add_argument('--dataset_src', type=str, required=True, help='Source YAML file for the dataset roots')
p.add_argument('--n_jobs', type=int, default=2, help='Number of jobs to run in parallel')
p.add_argument('--epochs', type=int, default=4, help='Number of epochs to run each cross-validation run')
p.add_argument('--cuda', action='store_true', help='Toggle to use CUDA')
p.add_argument('--opt_params', type=str, required=True, help='Source YAML file for optimizer params')
p = p.parse_args()

with open(p.params_src) as f:
    all_vals = yaml.safe_load(f)

n_layers = all_vals['nlayers']
net_type = all_vals['hidden_structure'][0]
# Generate correct values
for k in sorted(all_vals.keys()):

    # Hidden factors
    if k in ['hidden_factor1', 'hidden_factor2', 'dropout_probs']:
        if len(all_vals[k]) > 1:
            min_ = all_vals[k][0]
            max_ = all_vals[k][1]
            diff_ = all_vals[k][2]
            all_vals[k] = list(around(arange(min_, max_ + diff_, diff_).clip(min=min_, max=max_), 3))
        else:
            all_vals[k] = [None]

    # Dropout and Batch Normalization
    if k in ['dropout_probs', 'bnorms']:
        all_vals[k] = list(product(all_vals[k], repeat=n_layers[0]))

    all_vals['module__{}'.format(k)] = all_vals.pop(k)

# Load dataset
with open(p.dataset_src) as f:
    dset_src = yaml.safe_load(f)

if p.dataset == 'mnist':
    tr_, te_ = dset.get_MNIST_dataset(train_root_location=dset_src['train'], test_root_location=dset_src['test'])

elif p.dataset == 'cifar10':
    tr_, te_ = dset.get_CIFAR10_dataset(train_root_location=dset_src['train'], test_root_location=dset_src['test'])

elif p.dataset == 'svhn':
    tr_, te_ = dset.get_SVHN_dataset(train_root_location=dset_src['train'], test_root_location=dset_src['test'])

# Optimizer is set to Adam with learning rate = 3e-04
# Batch size is set to 64
# Criterion is set to cross_entropy

# This is a dummy
fcnet = mdl.FCNet(input_dim=100, hidden_structure='decreasing', hidden_factor1=0.8, hidden_factor2=0.5,
                  output_dim=10, nlayers=3, dropout_probs=[0, 0, 0], bnorms=[False, False, False],
                  init_scheme='xavier', activation='sigmoid', end_activation='softmax')

with open(p.opt_params, 'r') as f:
    opt_params = yaml.safe_load(f)

gsm = skm.grid_search_module(fcnet=fcnet, optimizer=opt_params,
                             criterion='cross_entropy', epochs=p.epochs, batch_size=64,
                             params=all_vals, use_cuda=p.cuda, n_jobs=p.n_jobs)
gsm.fit(tr_[0], tr_[1])

with open('best_params_{}_grid_search_{}.txt'.format(p.dataset, net_type), 'w') as f:
    f.write(str(gsm.best_params_))
