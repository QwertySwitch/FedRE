import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np


class Ensemble_orderFC(nn.Module):
    def __init__(self, in_features, out_features, num_models, first_layer=False,
                 bias=True, constant_init=False, p=0.5, random_sign_init=False,):
        super(Ensemble_orderFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(num_models, in_features))
        self.gamma = nn.Parameter(torch.Tensor(num_models, out_features))
        self.num_models = num_models
        self.random_sign_init = random_sign_init
        self.constant_init = constant_init
        self.probability = p
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_models, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.first_layer = first_layer

    def reset_parameters(self):
        nn.init.normal_(self.alpha, mean=1., std=0.1)
        nn.init.normal_(self.gamma, mean=1., std=0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def update_indices(self, indices):
        self.indices = indices

    def forward(self, x):
        if not self.training and self.first_layer:
            # Repeated pattern in test: [[A,B,C],[A,B,C]]
            x = torch.cat([x for i in range(self.num_models)], dim=0)

        num_examples_per_model = int(x.size(0) / self.num_models)
        extra = x.size(0) - (num_examples_per_model * self.num_models)
        # Repeated pattern: [[A,A],[B,B],[C,C]]
        if num_examples_per_model != 0:
            alpha = torch.cat(
                [self.alpha for i in range(num_examples_per_model)],
                dim=1).view([-1, self.in_features])
            gamma = torch.cat(
                [self.gamma for i in range(num_examples_per_model)],
                dim=1).view([-1, self.out_features])
            if self.bias is not None:
                bias = torch.cat(
                    [self.bias for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.out_features])
        else:
            alpha = self.alpha.clone()
            gamma = self.gamma.clone()
            if self.bias is not None:
                bias = self.bias.clone()
        if extra != 0:
            alpha = torch.cat([alpha, alpha[:extra]], dim=0)
            gamma = torch.cat([gamma, gamma[:extra]], dim=0)
            bias = torch.cat([bias, bias[:extra]], dim=0)

        result = self.fc(x*alpha)*gamma
        return result + bias if self.bias is not None else result


class Ensemble_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, first_layer=False, num_models=1, train_gamma=True,
                 bias=True, constant_init=False, p=0.5, random_sign_init=False):
        super(Ensemble_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(num_models, in_channels))
        self.train_gamma = train_gamma
        self.random_sign_init = random_sign_init
        self.constant_init = constant_init
        self.probability = p
        if train_gamma:
            self.gamma = nn.Parameter(torch.Tensor(num_models, out_channels))
        self.num_models = num_models
        if bias:
            #self.bias = nn.Parameter(torch.Tensor(self.out_channels))
            self.bias = nn.Parameter(torch.Tensor(self.num_models, out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.first_layer = first_layer

    def reset_parameters(self):
        nn.init.normal_(self.alpha, mean=1., std=0.1)
        nn.init.normal_(self.gamma, mean=1., std=0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def update_indices(self, indices):
        self.indices = indices

    def forward(self, x):
        if not self.training and self.first_layer:
            # Repeated pattern in test: [[A,B,C],[A,B,C]]
            x = torch.cat([x for i in range(self.num_models)], dim=0)
        if self.train_gamma:
            num_examples_per_model = int(x.size(0) / self.num_models)
            extra = x.size(0) - (num_examples_per_model * self.num_models)
            alpha = torch.cat(
                [self.alpha for i in range(num_examples_per_model)],
                dim=1).view([-1, self.in_channels])
            alpha.unsqueeze_(-1).unsqueeze_(-1)
            gamma = torch.cat(
                [self.gamma for i in range(num_examples_per_model)],
                dim=1).view([-1, self.out_channels])
            gamma.unsqueeze_(-1).unsqueeze_(-1)
            if self.bias is not None:
                bias = torch.cat(
                    [self.bias for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.out_channels])
                bias.unsqueeze_(-1).unsqueeze_(-1)

            if extra != 0:
                alpha = torch.cat([alpha, alpha[:extra]], dim=0)
                gamma = torch.cat([gamma, gamma[:extra]], dim=0)
                if self.bias is not None:
                    bias = torch.cat([bias, bias[:extra]], dim=0)
            result = self.conv(x*alpha)*gamma
            return result + bias if self.bias is not None else result
        else:
            num_examples_per_model = int(x.size(0) / self.num_models)
            alpha = torch.cat(
                [self.alpha for i in range(num_examples_per_model)],
                dim=1).view([-1, self.in_channels])
            alpha.unsqueeze_(-1).unsqueeze_(-1)

            if self.bias is not None:
                bias = torch.cat(
                    [self.bias for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.out_channels])
                bias.unsqueeze_(-1).unsqueeze_(-1)
            result = self.conv(x*alpha)
            return result + bias if self.bias is not None else result

