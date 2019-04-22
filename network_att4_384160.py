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
    
class STNLayer(nn.Module):
    def __init__(self, channel_in, multiply=True):
        super(STNLayer, self).__init__()
        c = channel_in
        C = c//32
        self.multiply = multiply
        self.conv_in = nn.Conv2d(c, C, kernel_size=1)
        self.conv_out = nn.Conv2d(C, 1, kernel_size=1)
        # Encoder
        self.conv1 = nn.Conv2d(C, 2*C, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(2*C)
        self.ReLU1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv2 = nn.Conv2d(2*C, 4*C, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(4*C)
        self.ReLU2 = nn.ReLU(True)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(4*C, 2*C, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(2*C)
        self.ReLU3 = nn.ReLU(True)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(2*C, C, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(C)
        self.ReLU4 = nn.ReLU(True)

    def forward(self, x):
        b, c, _, _ = x.size()
        #print("modules: x.shape: " + str(x.shape))
        y = self.conv_in(x)

        # Encode
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.ReLU1(y)
        size1 = y.size()
        y, indices1 = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.ReLU2(y)

        # Decode
        y = self.deconv1(y)
        y = self.bn3(y)
        y = self.ReLU3(y)
        y = self.unpool1(y,indices1,size1)
        y = self.deconv2(y)
        y = self.bn4(y)
        y = self.ReLU4(y)

        y = self.conv_out(y)
        #torch.save(y,'./STN_stage1.pkl')
        if self.multiply == True:
            return x * y
        else:
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

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        #res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.attc1 = ATTC(in_channels=1024, out_channels=2048)
        self.atts1 = ATTS(in_channels=1024, h=24, w=10)
        self.attc2 = ATTC(in_channels=1024, out_channels=2048)
        self.atts2 = ATTS(in_channels=1024, h=24, w=10)
        self.attc3 = ATTC(in_channels=1024, out_channels=2048)
        self.atts3 = ATTS(in_channels=1024, h=24, w=10)
        
        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(24, 10))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 10))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 10))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 10))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 10))

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
        x = self.backbone(x)
        
        att_c1 = self.attc1(x)
        att_s1 = self.atts1(x)
        att_c2 = self.attc2(x)
        att_s2 = self.atts2(x)
        att_c3 = self.attc3(x)
        att_s3 = self.atts3(x)
        
        att1 = att_c1 * (1.0 + att_s1)
        att2 = att_c2 * (1.0 + att_s2)
        att3 = att_c3 * (1.0 + att_s3)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        
        p1 = p1 * att1
        p2 = p2 * att2
        p3 = p3 * att3

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

