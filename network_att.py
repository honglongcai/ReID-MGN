import copy
import torch
import torch.nn as nn
from opt import opt
from torchvision.models.resnet import resnet50, resnet101, Bottleneck


class ATTC(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(ATTC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(in_channels, in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // reduction, out_channels),
                nn.Sigmoid()
                )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c )
        y = self.fc(y).view(b, self.out_channels, 1, 1)
        return y


class ATTS(nn.Module):
    def __init__(self, in_channels, h, w, reduction=8, acti='sig'):
        super(ATTS, self).__init__()
        self.r1 = reduction
        self.r2 = reduction ** 2
        self.acti = acti
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=3,
                               stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // reduction)
        self.relu1 = nn.ReLU(True)
        self.avgpool = nn.AdaptiveAvgPool2d((h, w))
        
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels // self.r2, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels // self.r2)
        self.relu2 = nn.ReLU(True)
        
        self.conv3 = nn.Conv2d(in_channels // self.r2, 1, kernel_size=3,
                               stride=1, padding=1)
        self.hardtanh = nn.Hardtanh()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.avgpool(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.conv3(y)
        if self.acti == 'hardtanh':
            y = self.hardtanh(y)
        elif self.acti == 'tanh':
            y = self.tanh(y)
        elif self.acti == 'sig':
            y = self.sigmoid(y)
        else:
            print('ERROR.......')
        return y
    
class MGN(nn.Module):
    def __init__(self):
        super(MGN, self).__init__()
        #num_classes = 751
        #num_classes = 4101
        num_classes = opt.cls_num
        feats = 256
        if opt.backbone == 'resnet50':
            resnet = resnet50(pretrained=True)
        elif opt.backbone == 'resnet101':
            resnet = resnet101(pretrained=True)

        self.backbone1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        )
        self.backbone2 = resnet.layer2
        self.backbone3 = resnet.layer3[0]
        self.attc = ATTC(in_channels=256, out_channels=512)
        self.atts = ATTS(in_channels=256, h=48, w=8)
        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 8))

        self.reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(self.reduction)

        self.fc_id_2048_0 = nn.Linear(feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        #print(x)
        #print('inside input size:', x.size())
        #print()
        x = self.backbone1(x)
        att_c = self.attc(x)
        att_s = self.atts(x)
        att = att_c * (1.0 + att_s)
        x = self.backbone2(x)
        print('x size:', x.size())
        print('att size:', att.size())
        x = x * att
        x = self.backbone3(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        fg_p1 = self.reduction(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction(z2_p3).squeeze(dim=3).squeeze(dim=2)

        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3

