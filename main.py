from __future__ import print_function
import numbers

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from networks.models import ModelFedCon, ModelFedConNot, Discriminator
import config as cf
import copy
import random
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import os
import sys
import time
import argparse
import datetime
import numpy as np
from networks import *
from torch.autograd import Variable
from utils import *
from torch.utils.data import DataLoader
import logging


parser = argparse.ArgumentParser(description='FedRE')
parser.add_argument('--lr', default=0.01, type=float, help='learning_rate')
parser.add_argument('--model', default='vgg9', type=str, help='model [vgg9]')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/svhn]')
parser.add_argument('--logdir', default='./logs', type=str, help='log file path')
parser.add_argument('--iid', default=1, type=int, help='dataset to iid or non-iid')
parser.add_argument('--num_users', default=10, type=int, help='The number of local clients')
parser.add_argument('--epoch', default=5, type=int, help='Number of epochs of each local models')
parser.add_argument('--round', default=50, type=int, help='Number of training rounds')
parser.add_argument('--out_dim', default=256, type=int, help='Feature output dimension')
parser.add_argument('--alg', default='fedavg-r1', type=str, help='Federated Algorithm = [fedavg/fedavg-r1]')
parser.add_argument('--device', type=int, default=0, help='The device to run the program')
parser.add_argument('--islocal', type=int, default=0, help='Federated Mode or Local Mode')
args = parser.parse_args()


ensemble_size=1
models = list()
dis = list()
train_loaders = list()
test_loaders = list()
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, args.epoch, cf.batch_size, cf.optim_type
batch_size=128
device = torch.device(args.device)

random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = True

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.as_tensor(image), torch.as_tensor(label)
    
args.logdir = args.logdir + ',' + args.alg

mkdirs(args.logdir)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
if args.iid == 0:
    id = 'non-iid'
else:
    id = 'iid'
    
file_list = os.listdir(args.logdir)
file_list2 = [file for file in file_list if file.startswith(f'{args.model}_{args.dataset}_{id}')]
count = len(file_list2)+1
log_file_name = f'{args.model}_{args.dataset}_{id}_{count}'
log_path=log_file_name+'.log'
logging.basicConfig(
    filename=os.path.join(args.logdir, log_path),
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

print('\n[Phase 1] : Data Preparation')

train_dataset, test_dataset, user_groups = get_dataset(args)

global_test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def train_val_test(dataset, idxs):
    idxs = list(idxs)
    random.shuffle(idxs)
    
    idxs_train = idxs[:int(0.8*len(idxs))]
    idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
    idxs_test = idxs[int(0.9*len(idxs)):]
    train = DatasetSplit(dataset, idxs_train)
    valid = DatasetSplit(dataset, idxs_val)
    test = DatasetSplit(dataset, idxs_test)
    return train, valid, test

def getNetwork(args):
    if args.dataset == 'cifar100':
        fcdim = 100
    else:
        fcdim = 10
    if 'r1' in args.alg: 
        net = ModelFedCon(args.model, args.out_dim, fcdim, num=1)
    else:
        net = ModelFedConNot(args.model, args.out_dim, fcdim)
    global_model = ModelFedCon(args.model, args.out_dim, fcdim, num=args.num_users)
    real_global_model = ModelFedConNot(args.model, args.out_dim, fcdim)
    for i in range(args.num_users):
        models.append(copy.deepcopy(net.to(device)))
        if args.iid == 1:
            t, v, te = train_val_test(train_dataset, user_groups[i])
            train_loaders.append(DataLoader(t, batch_size=128, shuffle=True))
            test_loaders.append(DataLoader(te, batch_size=128, shuffle=False))
        else:
            train_loaders.append(DataLoader(user_groups[i], batch_size=128, shuffle=True))
            test_loaders.append(DataLoader(user_groups[i], batch_size=128, shuffle=False))
    file_name = args.model

    return global_model.to(device), real_global_model.to(device), file_name

if 'r1' in args.alg:
    r = True
else:
    r = False

# Model
print('\n[Phase 2] : Model setup')
print(f'| Use Rank-1 Matrix : {r}')
logger.info(f'| Use Rank-1 Matrix : {r}')
print(f'| Building net type [{args.model}]...')
logger.info(f'| Building net type [{args.model}]...')
global_model, real_global_model, file_name = getNetwork(args)


criterion = nn.CrossEntropyLoss()
# Training

my_list = ['alpha', 'gamma', 'bias']



def train(epoch, model, trainloader_load):
    model.train()
    model.training = True
    train_loss = 0
    correct = 0
    total = 0
    if 'r1' in args.alg:
        params_multi_tmp = list(filter(lambda kv: (my_list[0] in kv[0]) or (my_list[1] in kv[0]) , model.named_parameters()))
        param_core_tmp = list(filter(lambda kv: (my_list[0] not in kv[0]) and (my_list[1] not in kv[0]), model.named_parameters()))
        params_multi = [param for name, param in params_multi_tmp]
        param_core = [param for name, param in param_core_tmp]
        optimizer = optim.SGD([
                    {'params': param_core,'weight_decay': 1e-4},
                    {'params': params_multi, 'weight_decay': 1e-4}
                ], lr=cf.learning_rate(args.lr, epoch), momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9)
    for ep in range(0, epoch):
        for batch_idx, (inputs, targets) in enumerate(trainloader_load):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            _, logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                    %(ep+1, num_epochs, batch_idx+1,
                        (len(trainloader_load.dataset)//batch_size)+1, loss.item(), 100.*correct/total))
            sys.stdout.flush()
        logger.info('| Epoch %d \t\tLoss: %.4f Acc@1: %.3f%%'
                    %(epoch, loss.item(), 100.*correct/total))
    if 'r1' in args.alg:
        params_multi = dict(filter(lambda kv: (my_list[0] in kv[0]) or (my_list[1] in kv[0]) or (my_list[2] in kv[0]), model.state_dict().items()))
        params_core = dict(filter(lambda kv: (my_list[0] not in kv[0]) and (my_list[1] not in kv[0]) and (my_list[2] not in kv[0]), model.state_dict().items()))
        return params_multi, params_core
    else:
        return None, model.state_dict()

def test(epoch, model, testloader):
    global best_acc
    model.eval()
    model.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            _, logits = model(inputs)
            loss = criterion(logits, targets)

            test_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100.*correct/total
        print("\n| Validation \t\t\t\tLoss: %.4f Acc@1: %.2f%%" %(loss.item(), acc))
        logger.info("\n| Validation \t\t\t\tLoss: %.4f Acc@1: %.2f%%" %(loss.item(), acc))
        
print('| Training Rounds = ' + str(args.round))
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

print('\n[Phase 3] : Training Clients')

update_model_ratio = 1

for round in range(args.round):
    elapsed_time = 0
    print('\n| Round [%3d/%3d]' %(round+1, args.round))
    logger.info('\n| Round [%3d/%3d]' %(round+1, args.round))
    local_weights = list()
    local_core_weights = list()
    for client in range(args.num_users):
        if round != 0 and 'r1' in args.alg:
            models[client].load_state_dict(replace_weight(models[client].state_dict(), core_layer_avg))
        if args.islocal != 1 and round != 0 and 'r1' not in args.alg:
            models[client].load_state_dict(core_layer_avg)
        print('\n=> Training Client #%d, LR=%.4f' %(client+1, args.lr))
        logger.info('\n=> Training Client #%d, LR=%.4f' %(client+1, args.lr))
        
        start_time = time.time()
        ensemble_layer, core_layer = train(num_epochs, models[client], train_loaders[client])
        local_weights.append(ensemble_layer)
        local_core_weights.append(core_layer)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        test(num_epochs, models[client], global_test_loader)
        print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))
    core_layer_avg = average_weights(local_core_weights)
    ensemble_layer_concat = concat_weights(local_weights)
    if (round + 1) % update_model_ratio == 0:
        global_model.load_state_dict(replace_weight(global_model.state_dict(), core_layer_avg))
        global_model.load_state_dict(replace_weight(global_model.state_dict(), ensemble_layer_concat))
        print('\n[Phase 4] : Testing Global Model')
        logger.info('\n[Phase 4] : Testing Global Model')
        test(10, global_model, global_test_loader)
        

    