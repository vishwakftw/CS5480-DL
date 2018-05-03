import yaml
import modules as mdl
import datasets as dset
import skorchModules as skm

from argparse import ArgumentParser as AP
from sklearn.metrics import accuracy_score

p = AP()
p.add_argument('--params_src', type=str, required=True, help='Source YAML file for the params')
p.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10', 'svhn'],
               help='Dataset name')
p.add_argument('--dataset_src', type=str, required=True, help='Source YAML file for the dataset roots')
p.add_argument('--epochs', type=int, default=4, help='Number of epochs to run each cross-validation run')
p.add_argument('--cuda', action='store_true', help='Toggle to use CUDA')
p.add_argument('--opt_params', type=str, required=True, help='Source YAML file for optimizer params')
p.add_argument('--logger', type=str, default=None, help='Path of the logger file')
p = p.parse_args()

# Load network params
with open(p.params_src, 'r') as f:
    net_params = yaml.safe_load(f)
    for k in net_params.keys():
        net_params[k.replace('module__', '')] = net_params.pop(k)

# Load optimizer params
with open(p.opt_params, 'r') as f:
    opt_params = yaml.safe_load(f)

# Load dataset
with open(p.dataset_src) as f:
    dset_src = yaml.safe_load(f)

if p.dataset == 'mnist':
    tr_, te_ = dset.get_MNIST_dataset(train_root_location=dset_src['train'], test_root_location=dset_src['test'])
    input_dim = 784
    output_dim = 10

elif p.dataset == 'cifar10':
    tr_, te_ = dset.get_CIFAR10_dataset(train_root_location=dset_src['train'], test_root_location=dset_src['test'])
    input_dim = 256
    output_dim = 10

elif p.dataset == 'svhn':
    tr_, te_ = dset.get_SVHN_dataset(train_root_location=dset_src['train'], test_root_location=dset_src['test'])
    input_dim = 192
    output_dim = 10

fcnet = mdl.FCNet(input_dim=input_dim, output_dim=output_dim, hidden_structure=net_params['hidden_structure'],
                  hidden_factor1=net_params['hidden_factor1'], hidden_factor2=net_params['hidden_factor2'],
                  nlayers=net_params['nlayers'], dropout_probs=net_params['dropout_probs'],
                  bnorms=net_params['bnorms'], init_scheme=net_params['init_scheme'],
                  activation=net_params['activation'], end_activation=net_params['end_activation'])

sk_clf = skm.get_generic_module(fcnet=fcnet, criterion='cross_entropy', optimizer=opt_params,
                                max_epochs=p.epochs, batch_size=64, use_cuda=p.cuda)
sk_clf.fit(tr_[0], tr_[1])
test_outputs = sk_clf.predict(te_[0])
test_accuracy = accuracy_score(te_[1], test_outputs)
print("Test accuracy obtained: {}".format(test_accuracy))

train_outputs = sk_clf.predict(tr_[0])
train_accuracy = accuracy_score(tr_[1], train_outputs)
print("Train accuracy obtained: {}".format(train_accuracy))

if p.logger is not None:
    with open(p.logger, 'a') as f:
        f.write('{}\t{}\t{}\t{}\n'.format(p.dataset, p.params_src, train_accuracy, test_accuracy))
