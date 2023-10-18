import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.fft

class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch, down=False):
        super(conv_block, self).__init__()
        if down:
            stride = 2
        else:
            stride = 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.SELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.SELU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class convfuse(nn.Module):
    """
    convfuse Block
    """
    def __init__(self, in_ch, out_ch):
        super(convfuse, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.In = nn.InstanceNorm2d(out_ch,affine=True,track_running_stats=True)
        self.selu = nn.SELU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        x = F.interpolate(x, y.size()[2:], mode='bilinear', align_corners=True)
        z = self.conv(x)
        z = self.bn(z)
        z = self.relu(z)
        return z


# 空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # 整个 ASPP 架构


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


# Fuse_Unet network
class Fuse_Unet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(Fuse_Unet, self).__init__()
        filters = [16, 32, 64, 128, 256, 512]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder1
        self.Conv1_1 = conv_block(input_nc, filters[0])
        self.Conv2_1 = conv_block(filters[0], filters[1])
        self.Conv3_1 = conv_block(filters[1], filters[2])
        # self.Conv4_1 = conv_block(filters[2], filters[3])

        self.Conv5_1 = ASPP(filters[2], [6, 12, 18], filters[3])
        
        # encoder2
        self.Conv1_2 = conv_block(input_nc, filters[0])
        self.Conv2_2 = conv_block(filters[0], filters[1])
        self.Conv3_2 = conv_block(filters[1], filters[2])
        # self.Conv4_2 = conv_block(filters[2], filters[3])

        self.Conv5_2 = ASPP(filters[2], [6, 12, 18], filters[3])

        # feature fusion
        self.Conv1 = convfuse(filters[1], filters[0])
        self.Conv2 = convfuse(filters[2], filters[1])
        self.Conv3 = convfuse(filters[3], filters[2])
        self.Conv4 = convfuse(filters[4], filters[3])
        # self.Conv1 = nn.Conv2d(filters[1], filters[0], kernel_size=3, stride=1, padding=1, bias=True)
        # self.Conv2 = nn.Conv2d(filters[2], filters[1], kernel_size=3, stride=1, padding=1, bias=True)
        # self.Conv3 = nn.Conv2d(filters[3], filters[2], kernel_size=3, stride=1, padding=1, bias=True)
        # self.Conv4 = nn.Conv2d(filters[4], filters[3], kernel_size=3, stride=1, padding=1, bias=True)

        # decoder
        self.Up_conv5 = conv_block(filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_nc, kernel_size=1, stride=1, padding=0)

    def encoder1(self, img_ir):

        # e2 = torch.cat((img_ir, img_vi), dim=1)
        e1_1 = self.Conv1_1(img_ir)

        e2_1 = self.Maxpool1(e1_1)
        e2_1 = self.Conv2_1(e2_1)

        e3_1 = self.Maxpool2(e2_1)
        e3_1 = self.Conv3_1(e3_1)

        e4_1 = self.Maxpool3(e3_1)
        # e4_1 = self.Conv4_1(e4_1)

        e4_1 = self.Conv5_1(e4_1)

        return e1_1,e2_1,e3_1,e4_1
        
    def encoder2(self, img_vi):
        img_vi = img_vi[:, :1]
        e1_2 = self.Conv1_2(img_vi)

        e2_2 = self.Maxpool1(e1_2)
        e2_2 = self.Conv2_2(e2_2)

        e3_2 = self.Maxpool2(e2_2)
        e3_2 = self.Conv3_2(e3_2)

        e4_2 = self.Maxpool3(e3_2)
        # e4_2 = self.Conv4_2(e4_2)

        e4_2 = self.Conv5_2(e4_2)

        return e1_2,e2_2,e3_2,e4_2


    def decoder(self, e1_1,e2_1,e3_1,e4_1,e1_2,e2_2,e3_2,e4_2):
        e1 = torch.cat((e1_1, e1_2), dim=1)
        e2 = torch.cat((e2_1, e2_2), dim=1)
        e3 = torch.cat((e3_1, e3_2), dim=1)
        e4 = torch.cat((e4_1, e4_2), dim=1)

        e1 = self.Conv1(e1)
        e2 = self.Conv2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Conv4(e4)



        d4 = self.Up_conv5(e4)

        d3 = self.Up4(d4, e3)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.Up_conv4(d3)

        d2 = self.Up3(d3, e2)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.Up_conv3(d2)

        d1 = self.Up2(d2, e1)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.Up_conv2(d1)

        out = self.Conv(d1)

        # d1 = self.active(out)

        return [out]
