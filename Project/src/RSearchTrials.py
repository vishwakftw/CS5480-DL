import yaml
import modules as mdl
import datasets as dset
import skorchModules as skm
import distributions as dist

from argparse import ArgumentParser as AP

p = AP()
p.add_argument('--params_src', type=str, required=True, help='Source YAML file for the params')
p.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10', 'svhn'],
               help='Dataset name')
p.add_argument('--dataset_src', type=str, required=True, help='Source YAML file for the dataset roots')
p.add_argument('--n_jobs', type=int, default=2, help='Number of jobs to run in parallel')
p.add_argument('--epochs', type=int, default=4, help='Number of epochs to run each cross-validation run')
p.add_argument('--n_iter', type=int, default=25, help='Number of random sampling iterations to perform')
p.add_argument('--cuda', action='store_true', help='Toggle to use CUDA')
p.add_argument('--opt_params', type=str, required=True, help='Source YAML file for optimizer params')
p = p.parse_args()

with open(p.params_src) as f:
    all_dist = yaml.safe_load(f)

n_layers = all_dist['nlayers']
net_type = all_dist['hidden_structure'][0]
# Generate correct values
for k in sorted(all_dist.keys()):

    # Hidden factors
    if k in ['hidden_factor1', 'hidden_factor2']:
        if len(all_dist[k]) > 1:
            min_ = all_dist[k][0]
            max_ = all_dist[k][1]
            diff_ = all_dist[k][2]
            all_dist[k] = dist.randominterval(low=min_, high=max_, diff=diff_, size=1)
        else:
            all_dist[k] = [None]

    # Dropout
    if k == 'dropout_probs':
        min_ = all_dist[k][0]
        max_ = all_dist[k][1]
        diff_ = all_dist[k][2]
        all_dist[k] = dist.randominterval(low=min_, high=max_, diff=diff_, size=n_layers[0])

    # Batch Normalization
    if k == 'bnorms':
        all_dist[k] = dist.randombool(size=n_layers[0])

    # Initialization scheme and activation function
    if k in ['init_scheme', 'activation']:
        all_dist[k] = dist.randomtext(phrases=all_dist[k], size=1)

    all_dist['module__{}'.format(k)] = all_dist.pop(k)

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

rsm = skm.random_search_module(fcnet=fcnet, optimizer=opt_params,
                               criterion='cross_entropy', epochs=p.epochs, batch_size=64,
                               param_dist=all_dist, use_cuda=p.cuda, n_jobs=p.n_jobs, n_iter=p.n_iter)
rsm.fit(tr_[0], tr_[1])

with open('best_params_{}_random_search_{}.txt'.format(p.dataset, net_type), 'w') as f:
    f.write(str(rsm.best_params_))
