#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import iid, noniid, noniid2
import pickle
import os
def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar10':
        data_dir = '../data/cifar/'
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
            std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                std=[x/255.0 for x in [63.0, 62.1, 66.7]]),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=False,
                                       transform=transform_train)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=False ,
                                      transform=transform_test)

        if args.iid:
            print('| Data Type = IID')
            user_groups = iid(train_dataset, args.num_users)
        else:
            print('| Data Type = Non-IID')
            user_groups = noniid2('CIFAR10', train_dataset, args.num_users)
            for i in range(args.num_users):
                user_groups[i].transform = transform_train
    if args.dataset == 'cifar100':
        data_dir = '../data/cifar/'
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
            std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                std=[x/255.0 for x in [63.0, 62.1, 66.7]]),
        ])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=transform_train)

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=transform_test)

        if args.iid:
            user_groups = iid(train_dataset, args.num_users)
        else:
            user_groups = noniid('CIFAR100', train_dataset, args.num_users)
    elif args.dataset == 'svhn':
        data_dir = '../data/svhn/'
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.SVHN(data_dir, split='train', download=True,
                                       transform=transform)

        test_dataset = datasets.SVHN(data_dir, split='test', download=True,
                                      transform=transform)
        if args.iid:
            user_groups = iid(train_dataset, args.num_users)
        else:
            user_groups = noniid('SVHN', train_dataset, args.num_users)
    elif args.dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        if args.iid:
            user_groups = iid(train_dataset, args.num_users)
        else:
            user_groups = noniid('MNIST', train_dataset, args.num_users)
    elif args.dataset == 'tinyimagenet':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        if args.iid:
            user_groups = iid(train_dataset, args.num_users)
        else:
            user_groups = noniid('TinyImageNet', train_dataset, args.num_users)
            
    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def replace_weight(w, r_w):
    for key2 in r_w.keys():
        w[key2] = r_w[key2]
    return w

def concat_weights(w):
    w_concat = copy.deepcopy(w[0])
    for key in w_concat.keys():
        for i in range(1, len(w)):
            w_concat[key] = torch.cat((w_concat[key], w[i][key]), dim=0)
    return w_concat

def replace_key(w):
    key_list = ['conv.', 'fc.']
    new_w = dict()
    for key in w.keys():
        if key_list[0] in key:
            new_key = key.replace(key_list[0], '')
            new_w[new_key] = w[key]
        if key_list[1] in key:
            new_key = key.replace(key_list[1], '')
            new_w[new_key] = w[key]
    return new_w

def matmul_layer(w, rs):
    """
    Returns the average of the weights.
    """
    key = ['.conv', '.fc']
    new_rs = dict()
    rskey = list(rs.keys())
    new_weight = dict()
    for n in range(0, len(rskey), 2):
        new_name = rskey[n].replace('.alpha', '')
        if 'linear' not in new_name and 'fc' not in new_name:
            a = rskey[n]
            b = rskey[n+1]
            rs_ab = rs[a] * rs[b].T
            rs_ab = rs_ab.unsqueeze(-1).unsqueeze(-1)
            new_w = w[new_name+f'{key[0]}.weight']
            new_w2 = torch.mul(new_w, rs_ab)
            new_weight[new_name+f'{key[0]}.weight'] = new_w2
        else:
            a = rskey[n]
            b = rskey[n+1]
            rs_ab = rs[a] * rs[b].T
            rs_ab = rs_ab
            new_w = w[new_name+f'{key[1]}.weight']
            new_w2 = torch.mul(new_w, rs_ab)
            new_weight[new_name+f'{key[1]}.weight'] = new_w2
    
    for key in new_weight.keys():
        w[key] = new_weight[key]
    
    return w

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
