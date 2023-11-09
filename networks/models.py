import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from networks.batchensemble_layers import *
import torchvision.models as models

class SimpleCNN_header(nn.Module):
    def __init__(self, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = Ensemble_Conv2d(3, 6, 5, first_layer=True, num_models=1, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = Ensemble_Conv2d(6, 16, 5, first_layer=False, num_models=1, bias=True)
        self.fc = Ensemble_orderFC(16 * 5 * 5, output_dim, num_models=1)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1 , 16 * 5 * 5)
        z = self.fc(x)
        return z

class SimpleCNN_header_Not(nn.Module):
    def __init__(self, output_dim=10):
        super(SimpleCNN_header_Not, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc = nn.Linear(16 * 5 * 5, output_dim)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1 , 16 * 5 * 5)
        z = self.fc(x)
        return z

class ModelFedCon(nn.Module):
    def __init__(self, base_model, out_dim, fc_dim, num):
        super(ModelFedCon, self).__init__()
        pro_dim = 256
        self.num = num
        self.fc_dim = fc_dim
        if base_model == 'simple-cnn':
            self.basemodel = SimpleCNN_header(output_dim=out_dim)
            self.fc2 = Ensemble_orderFC(out_dim, fc_dim, num_models=num)
        elif base_model == 'vgg9':
            self.basemodel = VGG(make_layers(cfg['A'], num_models=num), output_dim=out_dim, num_models=num)
            self.fc2 = Ensemble_orderFC(out_dim, fc_dim, num_models=num, bias=True)
        elif base_model == 'vgg16':
            self.basemodel = VGG(make_layers(cfg['A'], num_models=num), output_dim=out_dim, num_models=num)
            self.fc2 = Ensemble_orderFC(out_dim, fc_dim, num_models=num, bias=True)
        elif base_model == 'resnet18':
            self.basemodel = ResNet(BasicBlock, [2, 2, 2, 2], out_dim)
            self.fc2 = Ensemble_orderFC(out_dim, fc_dim, num_models=num, bias=True)
        elif base_model == 'resnet152':
            self.basemodel = ResNet(Bottleneck, [3, 8, 36, 3], out_dim)
            self.fc2 = Ensemble_orderFC(out_dim, fc_dim, num_models=num, bias=True)
    def forward(self, x):
        feature = self.basemodel(x)
        prediction = self.fc2(feature)
        if self.num != 1:
            prediction = F.softmax(prediction, dim=1)
            prediction = prediction.view([self.num, -1, self.fc_dim]).mean(dim=0)
        return feature, prediction

class ModelFedConNot(nn.Module):
    def __init__(self, base_model, out_dim, fc_dim):
        super(ModelFedConNot, self).__init__()
        pro_dim = 256
        if base_model == 'simple-cnn':
            self.basemodel = SimpleCNN_header_Not(output_dim=out_dim)
            self.fc2 = nn.Linear(out_dim, fc_dim)
        elif base_model == 'vgg9':
            self.basemodel = VGG_not(make_layers_not(cfg['A']), output_dim=out_dim)
            self.fc2 = nn.Linear(out_dim, fc_dim)
        elif base_model == 'vgg16':
            self.basemodel = VGG_not(make_layers_not(cfg['A']), output_dim=out_dim)
            self.fc2 = nn.Linear(out_dim, fc_dim)
        elif base_model == 'resnet18':
            self.basemodel = ResNetNot(BasicBlockNot, [2, 2, 2, 2], out_dim)
            self.fc2 = nn.Linear(out_dim, fc_dim)
        elif base_model == 'resnet152':
            self.basemodel = ResNetNot(BottleneckNot, [3, 8, 36, 3], out_dim)
            self.fc2 = nn.Linear(out_dim, fc_dim)
    def forward(self, x):
        feature = self.basemodel(x)
        prediction = self.fc2(feature)
        
        return feature, prediction

class VGG(nn.Module):
    def __init__(self, features, output_dim, num_models):
        super(VGG, self).__init__()
        self.features = features
        self.linear = Ensemble_orderFC(512, output_dim, first_layer=False, num_models=num_models, bias=True)
        for m in self.modules():
            if isinstance(m, Ensemble_Conv2d):
                n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.out_channels
                m.conv.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def make_layers(cfg, num_models,batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if i == 0:
                conv2d = Ensemble_Conv2d(in_channels, v, kernel_size=3, padding=1, first_layer=True, num_models=num_models, bias=True)
            else:
                conv2d = Ensemble_Conv2d(in_channels, v, kernel_size=3, padding=1, first_layer=False, num_models=num_models, bias=True)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG_not(nn.Module):
    def __init__(self, features, output_dim):
        super(VGG_not, self).__init__()
        self.features = features
        self.linear = nn.Linear(512, output_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def make_layers_not(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        

    def forward(self, x):
        validity = self.model(x)
        return validity


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg9():
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Ensemble_Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Ensemble_Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Ensemble_Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Ensemble_Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Ensemble_Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Ensemble_Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Ensemble_Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlockNot(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockNot, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckNot(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckNot, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = Ensemble_Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, first_layer=True, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = Ensemble_orderFC(512*block.expansion, num_classes, num_models=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetNot(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetNot, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])