"""
Modify the original file to make the class support feature extraction
"""
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv1x3x3(in_planes, out_planes, stride=1, t_stride=1):
    """1x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3),
                     stride=(t_stride, stride, stride),
                     padding=(0, 1, 1), bias=False)

def conv3x1x1(in_planes, out_planes, stride=1, t_stride=1):
    """3x1x1 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1),
                     stride=(t_stride, stride, stride),
                     padding=(1, 0, 0), bias=False)


class Bottleneck3D_11113(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(Bottleneck3D_11113, self).__init__()


        self.conv1_1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), bias=False)
        self.bn1_1 = nn.BatchNorm3d(planes)

        self.conv1_2 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), bias=False)
        self.bn1_2 = nn.BatchNorm3d(planes)

        self.conv1_3 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), bias=False)
        self.bn1_3 = nn.BatchNorm3d(planes)

        self.conv1_4 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), bias=False)
        self.bn1_4 = nn.BatchNorm3d(planes)

        self.conv1_5 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), bias=False)
        self.bn1_5 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3),
                               stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        print("r1.shape:", residual.shape)
        x1 = x[:, :, 0:1, :, :]
        out1 = self.conv1_1(x1)
        out1 = self.bn1_1(out1)
        out1 = self.relu(out1)

        x2 = x[:, :, 1:3, :, :]
        out2 = self.conv1_2(x2)
        out2 = self.bn1_2(out2)
        out2 = self.relu(out2)

        x3 = x[:, :, 3:7, :, :]
        out3 = self.conv1_3(x3)
        out3 = self.bn1_3(out3)
        out3 = self.relu(out3)

        x4 = x[:, :, 7:15, :, :]
        out4 = self.conv1_4(x4)
        out4 = self.bn1_4(out4)
        out4 = self.relu(out4)

        x5 = x[:, :, 15:31, :, :]
        out5 = self.conv1_5(x5)
        out5 = self.bn1_5(out5)
        out5 = self.relu(out5)

        out = torch.cat([out1, out2, out3, out4, out5], dim=2)
        print('out_1.shape:', out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        print('out_2.shape:', out.shape)
        out = self.conv3(out)
        out = self.bn3(out)
        print("out.shape:", out.shape)

        if self.downsample is not None:
            residual = self.downsample(x)
            print("r2.shape:", residual.shape)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck3D_11133(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(Bottleneck3D_11133, self).__init__()


        self.conv1_1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), bias=False)
        self.bn1_1 = nn.BatchNorm3d(planes)

        self.conv1_2 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), bias=False)
        self.bn1_2 = nn.BatchNorm3d(planes)

        self.conv1_3 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), bias=False)
        self.bn1_3 = nn.BatchNorm3d(planes)

        self.conv1_4 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), bias=False)
        self.bn1_4 = nn.BatchNorm3d(planes)

        self.conv1_5 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), bias=False)
        self.bn1_5 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3),
                               stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        x1 = x[:, :, 0:1, :, :]
        out1 = self.conv1_1(x1)
        out1 = self.bn1_1(out1)
        out1 = self.relu(out1)

        x2 = x[:, :, 1:3, :, :]
        out2 = self.conv1_2(x2)
        out2 = self.bn1_2(out2)
        out2 = self.relu(out2)

        x3 = x[:, :, 3:7, :, :]
        out3 = self.conv1_3(x3)
        out3 = self.bn1_3(out3)
        out3 = self.relu(out3)

        x4 = x[:, :, 7:15, :, :]
        out4 = self.conv1_4(x4)
        out4 = self.bn1_4(out4)
        out4 = self.relu(out4)

        x5 = x[:, :, 15:31, :, :]
        out5 = self.conv1_5(x5)
        out5 = self.bn1_5(out5)
        out5 = self.relu(out5)

        out = torch.cat([out1, out2, out3, out4, out5], dim=2)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck3D_11333(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(Bottleneck3D_11333, self).__init__()


        self.conv1_1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), bias=False)
        self.bn1_1 = nn.BatchNorm3d(planes)

        self.conv1_2 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), bias=False)
        self.bn1_2 = nn.BatchNorm3d(planes)

        self.conv1_3 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), bias=False)
        self.bn1_3 = nn.BatchNorm3d(planes)

        self.conv1_4 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), bias=False)
        self.bn1_4 = nn.BatchNorm3d(planes)

        self.conv1_5 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), bias=False)
        self.bn1_5 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3),
                               stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        x1 = x[:, :, 0:1, :, :]
        out1 = self.conv1_1(x1)
        out1 = self.bn1_1(out1)
        out1 = self.relu(out1)

        x2 = x[:, :, 1:3, :, :]
        out2 = self.conv1_2(x2)
        out2 = self.bn1_2(out2)
        out2 = self.relu(out2)

        x3 = x[:, :, 3:7, :, :]
        out3 = self.conv1_3(x3)
        out3 = self.bn1_3(out3)
        out3 = self.relu(out3)

        x4 = x[:, :, 7:15, :, :]
        out4 = self.conv1_4(x4)
        out4 = self.bn1_4(out4)
        out4 = self.relu(out4)

        x5 = x[:, :, 15:31, :, :]
        out5 = self.conv1_5(x5)
        out5 = self.bn1_5(out5)
        out5 = self.relu(out5)

        out = torch.cat([out1, out2, out3, out4, out5], dim=2)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck3D_13333(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(Bottleneck3D_13333, self).__init__()


        self.conv1_1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), bias=False)
        self.bn1_1 = nn.BatchNorm3d(planes)

        self.conv1_2 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), bias=False)
        self.bn1_2 = nn.BatchNorm3d(planes)

        self.conv1_3 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), bias=False)
        self.bn1_3 = nn.BatchNorm3d(planes)

        self.conv1_4 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), bias=False)
        self.bn1_4 = nn.BatchNorm3d(planes)

        self.conv1_5 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), bias=False)
        self.bn1_5 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3),
                               stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        x1 = x[:, :, 0:1, :, :]
        out1 = self.conv1_1(x1)
        out1 = self.bn1_1(out1)
        out1 = self.relu(out1)

        x2 = x[:, :, 1:3, :, :]
        out2 = self.conv1_2(x2)
        out2 = self.bn1_2(out2)
        out2 = self.relu(out2)

        x3 = x[:, :, 3:7, :, :]
        out3 = self.conv1_3(x3)
        out3 = self.bn1_3(out3)
        out3 = self.relu(out3)

        x4 = x[:, :, 7:15, :, :]
        out4 = self.conv1_4(x4)
        out4 = self.bn1_4(out4)
        out4 = self.relu(out4)

        x5 = x[:, :, 15:31, :, :]
        out5 = self.conv1_5(x5)
        out5 = self.bn1_5(out5)
        out5 = self.relu(out5)

        out = torch.cat([out1, out2, out3, out4, out5], dim=2)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet3D(nn.Module):

    def __init__(self, block, layers, num_classes=1000, feat=False, lite=False, **kwargs):
        if not isinstance(block, list):
            block = [block] * 4
        else:
            assert(len(block)) == 4, "Block number must be 4 for ResNet-Stype networks."
        self.inplanes = 64
        super(ResNet3D, self).__init__()
        self.feat = feat
        self.conv1_1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7),
                               stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        self.bn1_1 = nn.BatchNorm3d(64)

        self.conv1_2 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7),
                               stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        self.bn1_2 = nn.BatchNorm3d(64)

        self.conv1_3 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                               stride=(1, 2, 2), padding=(1, 3, 3),
                               bias=False)
        self.bn1_3 = nn.BatchNorm3d(64)

        self.conv1_4 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                               stride=(1, 2, 2), padding=(1, 3, 3),
                               bias=False)
        self.bn1_4 = nn.BatchNorm3d(64)

        self.conv1_5 = nn.Conv3d(3, 64, kernel_size=(5, 7, 7),
                               stride=(1, 2, 2), padding=(2, 3, 3),
                               bias=False)
        self.bn1_5 = nn.BatchNorm3d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block[3], 512, layers[3], stride=2)
        self.avgpool_1 = nn.AvgPool3d(kernel_size=(1, 7, 7), stride=1)
        self.avgpool_2 = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1)
        self.avgpool_3 = nn.AvgPool3d(kernel_size=(4, 7, 7), stride=1)
        self.avgpool_4 = nn.AvgPool3d(kernel_size=(8, 7, 7), stride=1)
        self.avgpool_5 = nn.AvgPool3d(kernel_size=(16, 7, 7), stride=1)
        self.feat_dim = 512 * block[0].expansion
        if not feat:
            self.fc = nn.Linear(512 * block[0].expansion * 5, num_classes)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                #if block[0] is BasicBlockSTF_Residual and "_2" in n:

                #    nn.init.constant_(m.weight, 0)
                #else:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, t_stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(t_stride, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, t_stride=t_stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        print('x.shape:', x.shape)
        x1 = x[:, :, ::16, :, :]
        print("x1.shape", x1.shape)
        x1 = self.conv1_1(x1)
        x1 = self.bn1_1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        print("x1.shape", x1.shape)

        x2 = x[:, :, ::8, :, :]
        print("x2.shape", x2.shape)
        x2 = self.conv1_2(x2)
        x2 = self.bn1_2(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        print("x2.shape", x2.shape)

        x3 = x[:, :, ::4, :, :]
        x3 = self.conv1_3(x3)
        x3 = self.bn1_3(x3)
        x3 = self.relu(x3)
        x3 = self.maxpool(x3)

        x4 = x[:, :, ::2, :, :]
        x4 = self.conv1_4(x4)
        x4 = self.bn1_4(x4)
        x4 = self.relu(x4)
        x4 = self.maxpool(x4)

        x5 = x[:, :, :, :, :]
        print("x5.shape", x5.shape)
        x5 = self.conv1_5(x5)
        x5 = self.bn1_5(x5)
        x5 = self.relu(x5)
        x5 = self.maxpool(x5)
        print("x5.shape", x5.shape)

        x = torch.cat([x1, x2, x3, x4, x5], dim=2)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = x[:, :, 0:1, :, :]
        x2 = x[:, :, 1:3, :, :]
        x3 = x[:, :, 3:7, :, :]
        x4 = x[:, :, 7:15, :, :]
        x5 = x[:, :, 15:31, :, :]

        x1 = self.avgpool_1(x1)
        x2 = self.avgpool_2(x2)
        x3 = self.avgpool_3(x3)
        x4 = self.avgpool_4(x4)
        x5 = self.avgpool_5(x5)


        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = x.view(x.size(0), -1)
        if not self.feat:
            print("WARNING!!!!!!!")
            x = self.fc(x)
        return x


def part_state_dict(state_dict, model_dict):
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    pretrained_dict = inflate_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)
    return model_dict


def inflate_state_dict(pretrained_dict, model_dict):
    for k in pretrained_dict.keys():
        if pretrained_dict[k].size() != model_dict[k].size():
            assert(pretrained_dict[k].size()[:2] == model_dict[k].size()[:2]), \
                   "To inflate, channel number should match."
            assert(pretrained_dict[k].size()[-2:] == model_dict[k].size()[-2:]), \
                   "To inflate, spatial kernel size should match."
            print("Layer {} needs inflation.".format(k))
            shape = list(pretrained_dict[k].shape)
            shape.insert(2, 1)
            t_length = model_dict[k].shape[2]
            pretrained_dict[k] = pretrained_dict[k].reshape(shape)
            if t_length != 1:
                pretrained_dict[k] = pretrained_dict[k].expand_as(model_dict[k]) / t_length
            assert(pretrained_dict[k].size() == model_dict[k].size()), \
                   "After inflation, model shape should match."

    return pretrained_dict

def resnet50_3d(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet3D([Bottleneck3D_000, Bottleneck3D_000, Bottleneck3D_100, Bottleneck3D_100],
                     [3, 4, 6, 3], feat=feat, **kwargs)
    # import pdb
    # pdb.set_trace()
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def resnet50_3d_lite(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet3D([Bottleneck3D_000, Bottleneck3D_000, Bottleneck3D_000, Bottleneck3D_100],
                     [3, 4, 6, 3], feat=feat, lite=True, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def resnet50_mgn(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet3D([Bottleneck3D_11113, Bottleneck3D_11133, Bottleneck3D_11333, Bottleneck3D_13333],
                     [3, 4, 6, 3], feat=feat, **kwargs)
    # import pdb
    # pdb.set_trace()
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it.
    net = resnet50_mgn()
    print(net)
    input = (torch.FloatTensor(8, 3, 16, 224, 224))
    output = net(input)
    print('net output size:')
    print(output.shape)
