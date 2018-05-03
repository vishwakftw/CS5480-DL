import yaml
import operator
import numpy as np
import torch.nn as nn
import datasets as dset

from argparse import ArgumentParser as AP
from sklearn.metrics import accuracy_score
from skorchModules import RandomNNetClassifier as RNC


p = AP()
p.add_argument('--n_classifiers', type=int, default=3, help='Number of classifier to take for an ensemble')
p.add_argument('--weights', type=str, default='uniform', choices=['uniform', 'dropout', 'batchnorm', 'nlayers'],
               help='Choice for weighing scheme:\n\
                     uniform: All classifiers weighted equally\n\
                     dropout: Classifiers with dropout are weighted higher\n\
                     batchnorm: Classifiers with batchnorm are weighted higher\n\
                     nlayers: Classifier with more layers are weighted higher')
p.add_argument('--voting', type=str, default='hard', choices=['hard', 'soft'], help='Voting scheme: hard / soft')
p.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10', 'svhn'], help='Dataset')
p.add_argument('--dataset_src', type=str, required=True, help='YAML file with the source of the dataset')
p.add_argument('--ntrials', type=int, default=5, help='Number of trials to conducted to average out the results')
p.add_argument('--cuda', action='store_true', help='Toggle to use CUDA')
p.add_argument('--opt_params', type=str, required=True, help='Source YAML file for optimizer params')
p.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs to train each classifier')
p.add_argument('--logger', type=str, default=None, help='Path of the logger file')
p.add_argument('--shownets', action='store_true', help='Display the architectures of the nets')
p = p.parse_args()


# Load optimizer params
with open(p.opt_params, 'r') as f:
    opt_params = yaml.safe_load(f)

# Load dataset
with open(p.dataset_src) as f:
    dset_src = yaml.safe_load(f)

if p.dataset == 'cifar10':
    input_dim = 256
    output_dim = 10
    tr_, te_ = dset.get_CIFAR10_dataset(train_root_location=dset_src['train'], test_root_location=dset_src['test'])

if p.dataset == 'svhn':
    input_dim = 192
    output_dim = 10
    tr_, te_ = dset.get_SVHN_dataset(train_root_location=dset_src['train'], test_root_location=dset_src['test'])

if p.dataset == 'mnist':
    input_dim = 784
    output_dim = 10
    tr_, te_ = dset.get_MNIST_dataset(train_root_location=dset_src['train'], test_root_location=dset_src['test'])


# Create the classifiers and fit on data
train_accus = []
test_accus = []
for t in range(0, p.ntrials):
    clfs = []
    for i in range(0, p.n_classifiers):
        clf = RNC(input_dim=input_dim, output_dim=output_dim, end_activation='softmax',
                  optimizer=opt_params, use_cuda=p.cuda, max_epochs=p.max_epochs)
        clf.fit(tr_[0], tr_[1])
        clfs.append(clf)


    # Get weights
    if p.weights == 'uniform':
        wghts = [round(1 / p.n_classifiers, 5) for i in range(0, p.n_classifiers)]

    elif p.weights == 'dropout':
        wghts = []
        for i in range(p.n_classifiers):
            count = 0
            for j in list(clfs[i].module_._modules.values()):
                count = (count + 1) if isinstance(j, nn.Dropout) else count
            wghts.append(count)
        if sum(wghts) == 0:
            wghts = [round(1 / p.n_classifiers, 5) for i in range(0, p.n_classifiers)]
        else:
            wghts = [k / sum(wghts) for k in wghts]

    elif p.weights == 'batchnorm':
        wghts = []
        for i in range(p.n_classifiers):
            count = 0
            for j in list(clfs[i].module_._modules.values()):
                count = (count + 1) if isinstance(j, nn.BatchNorm1d) else count
            wghts.append(count)
        if sum(wghts) == 0:
            wghts = [round(1 / p.n_classifiers, 5) for i in range(0, p.n_classifiers)]
        else:
            wghts = [k / sum(wghts) for k in wghts]

    elif p.weights == 'nlayers':
        wghts = []
        for i in range(p.n_classifiers):
            wghts.append(clfs[i].module_.nlayers)
        wghts = [k / sum(wghts) for k in wghts]


    # Predict the classes
    test_outputs = None
    train_outputs = None
    for i in range(0, p.n_classifiers):
        if test_outputs is None:
            test_outputs = wghts[i] * clf.predict_proba(te_[0])
            train_outputs = wghts[i] * clf.predict_proba(tr_[0])
        else:
            test_outputs += wghts[i] * clf.predict_proba(te_[0])
            train_outputs += wghts[i] * clf.predict_proba(tr_[0])
        clf.module_.net_init()  # Reinitialize module for next round

    if p.voting == 'hard':
        test_outputs = np.apply_along_axis(lambda x: max(enumerate(x),
                                           key=operator.itemgetter(1))[0], axis=1, arr=test_outputs)
        train_outputs = np.apply_along_axis(lambda x: max(enumerate(x),
                                            key=operator.itemgetter(1))[0], axis=1, arr=train_outputs)

    if p.voting == 'soft':
        test_outputs = np.argmax(test_outputs, axis=1)
        train_outputs = np.argmax(train_outputs, axis=1)

    test_accuracy = accuracy_score(te_[1], test_outputs)
    print("Test accuracy obtained: {} for run {}".format(test_accuracy, t))

    train_accuracy = accuracy_score(tr_[1], train_outputs)
    print("Train accuracy obtained: {} for run {}".format(train_accuracy, t))

    test_accus.append(test_accuracy)
    train_accus.append(train_accuracy)

avg_train_accuracy = sum(train_accus) / len(train_accus)
avg_test_accuracy = sum(test_accus) / len(test_accus)

if p.logger is not None:
    with open(p.logger, 'a') as f:
        f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(p.dataset, p.n_classifiers, p.voting, p.weights,
                                                  avg_train_accuracy, avg_test_accuracy))

if p.shownets:
    for i, clf in enumerate(clfs):
        print("Classifier {}\n{}".format(i, repr(clf.module_)))
