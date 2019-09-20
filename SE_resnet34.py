# coding:utf8
#from .basic_module import BasicModule
from torch import nn
from torch.nn import functional as F
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel))
        self.right = shortcut
        #SE Layers,Use nn.Conv2d instead of nn.Linear
        self.fc1 = nn.Conv2d(outchannel, outchannel//16, kernel_size=1)
        self.fc2 = nn.Conv2d(outchannel//16, outchannel, kernel_size=1)

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        #squeeze
        w = F.avg_pool2d(out,out.size(2))#beatch_size*1*1*c
        # print(w.size())#[32, 128, 1, 1]*2,[32, 256, 1, 1]*4,[32, 512, 1, 1]*9
        w = self.fc1(w)#beatch_size*1*1*(c/16)
        w = F.relu(w)#beatch_size*1*1*(c/16)
        w = self.fc2(w)#beatch_size*1*1*(c)
        w = F.sigmoid(w)#beatch_size*1*1*(c)
        #Excitation
        out = out * w
        out += residual
        return F.relu(out)


class SE_ResNet34(nn.Module):
    def __init__(self, num_classes=8):
        super(SE_ResNet34, self).__init__()
        self.model_name = 'resnet34'

        # 前几层: 图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))

        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel))
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        # print(x.size())#[32, 64, 128, 128]
        x = self.layer1(x)
        # print(x.size())#[32, 128, 128, 128]
        x = self.layer2(x)
        # print(x.size())#[32, 256, 64, 64]
        x = self.layer3(x)
        # print(x.size())#[32, 512, 32, 32]
        x = self.layer4(x)
        # print(x.size())#[32, 512, 16, 16]

        x = F.avg_pool2d(x, 7)
        # print(x.size())#[32, 512, 1, 1]
        x = x.view(x.size(0), -1)
        return self.fc(x)
