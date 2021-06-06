

import torch
import torch.nn as nn
from model.module import *
from torchvision import ops
import numpy as np
from skimage import segmentation
# import os
import torch
import torch.nn as nn
class MsG(nn.Module):
    def __init__(self, mode):
        super(MsG, self).__init__()
        assert mode in {'N', 'C', 'I'}
        self.mode = mode
        self.guidenet = Net()
        self.normalnet = Net1()
        if mode == 'C':

            self.predict_normal6 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal4 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.predict_normal2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True)

            self.conv_sparse1 = ResBlock(channels_in=2, num_filters=32, stride=1)  ####################################3
            self.conv_sparse2 = ResBlock(channels_in=32, num_filters=64, stride=1)
            self.conv_sparse3 = ResBlock(channels_in=64, num_filters=128, stride=2)
            self.conv_sparse4 = ResBlock(channels_in=128, num_filters=128, stride=2)
            self.conv_sparse5 = ResBlock(channels_in=128, num_filters=256, stride=2)
            self.conv_sparse6 = ResBlock(channels_in=256, num_filters=256, stride=2)

            self.conv_rgb1 = ResBlock(channels_in=3, num_filters=32, stride=1)
            self.conv_rgb2 = ResBlock(channels_in=32, num_filters=64, stride=1)
            self.conv_rgb3 = ResBlock(channels_in=64, num_filters=128, stride=2)
            self.conv_rgb3_1 = ResBlock(channels_in=128, num_filters=128, stride=1)
            self.conv_rgb4 = ResBlock(channels_in=128, num_filters=128, stride=2)
            self.conv_rgb4_1 = ResBlock(channels_in=128, num_filters=128, stride=1)
            self.conv_rgb5 = ResBlock(channels_in=128, num_filters=256, stride=2)
            self.conv_rgb5_1 = ResBlock(channels_in=256, num_filters=256, stride=1)
            self.conv_rgb6 = ResBlock(channels_in=256, num_filters=256, stride=2)
            self.conv_rgb6_1 = ResBlock(channels_in=256, num_filters=256, stride=1)

            self.predict_mask = nn.Sequential(
                nn.Conv2d(97, 1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Sigmoid()
            )

            self.deconv6 = nn.ConvTranspose2d(256 + 256 + 1, 256, 4, 2, 1, bias=False)  ###使用的是逆卷积操作卷积操作
            self.deconv5 = nn.ConvTranspose2d(256 + 256 + 1, 256, 4, 2, 1, bias=False)
            self.deconv4 = nn.ConvTranspose2d(256 + 128 + 1, 128, 4, 2, 1, bias=False)
            self.deconv3 = nn.ConvTranspose2d(128 + 128 + 1, 128, 4, 2, 1, bias=False)
            self.inconv2 = ResBlock(channels_in=128 + 64, num_filters=97, stride=1)    #3########no  64  the next time !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.predict = nn.Conv2d(97, 2, kernel_size=3, stride=1, padding=1, bias=True)

        self.normalconv=uncertainty_net()
        self.normalconv1 = uncertainty_net1()
        self.offset = nn.Conv2d(97, 18, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_offset=ops.DeformConv2d(in_channels=2,out_channels=2,kernel_size=3,padding=1)

    def forward(self, rgb, lidar, mask):
        b, c, w, h = rgb.size()

        if self.mode=='I':
            guide = torch.cat((rgb, lidar), 1)
            embedding0, em1, em2 = self.guidenet(guide)
            global_features = embedding0[:, 0:1, :, :]
            dense = self.normalconv(lidar+global_features, em1, em2)


        elif self.mode=='N':

            dense1,a,b=self.normalnet(torch.cat((mask,rgb),1))##
            dense=self.normalconv1(dense1+lidar,a,b)

        else:

            sparse_input = torch.cat((lidar, mask), 1)

            s1 = self.conv_sparse1(sparse_input)  # 256 x 512
            s2 = self.conv_sparse2(s1)  # 128 x 256
            s3 = self.conv_sparse3(s2)  # 64 x 128
            s4 = self.conv_sparse4(s3)  # 32 x 64
            s5 = self.conv_sparse5(s4)  # 16 x 32
            s6 = self.conv_sparse6(s5)  # 8 x 16

            r1 = self.conv_rgb1(rgb)  # 256 x 512
            r2 = self.conv_rgb2(r1)  # 128 x 256
            r3 = self.conv_rgb3_1(self.conv_rgb3(r2))  # 64 x 128
            r4 = self.conv_rgb4_1(self.conv_rgb4(r3))  # 32 x 64
            r5 = self.conv_rgb5_1(self.conv_rgb5(r4))  # 16 x 32
            r6 = self.conv_rgb6_1(self.conv_rgb6(r5)) + s6  # 8 x 16

            br3 = self.predict_normal3(r3)
            br4 = self.predict_normal4(r4)
            br5 = self.predict_normal5(r5)
            br6 = self.predict_normal6(r6)
            up5 = self.deconv6(adaptive_cat(s6 + r6, s6, br6))
            up4 = self.deconv5(adaptive_cat(s5 + r5, up5, br5))
            up3 = self.deconv4(adaptive_cat(s4 + r4, up4, br4))
            up2 = self.deconv3(adaptive_cat(s3 + r3, up3, br3))
            cat = self.inconv2(torch.cat((up2, s2), 1))######################
            dense = self.predict(cat)

        if self.mode == 'I':
            dense = F.interpolate(dense, (w, h), mode='bilinear', align_corners=True)
            return dense,global_features
        if self.mode == 'C':
            color_path_mask = self.predict_mask(cat)
            offset = self.offset(cat)
            dense = self.conv_offset(dense, offset)
            return dense, color_path_mask, cat#
        if self.mode == 'N':
            return dense#

class maskBlock(nn.Module):
    def __init__(self):
        super(maskBlock, self).__init__()
        self.mask_block = self.make_layers()

    def make_layers(self):
        in_channels = 97
        cfg = [97, 97]


        out_channels = 1
        layers = []

        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=True)

            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]

            in_channels = v

        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.mask_block(x)


class maskBlock1(nn.Module):
    def __init__(self):
        super(maskBlock1, self).__init__()
        self.mask_block = self.make_layers()

    def make_layers(self):
        in_channels = 2
        cfg = [2]

        out_channels = 1
        layers = []

        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=True)

            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]

            in_channels = v

        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.mask_block(x)

class deepMsG(nn.Module):
    def __init__(self):
        super(deepMsG, self).__init__()
        self.normal = MsG(mode='I')
        self.color_path = MsG(mode='C')
        self.normal_path = MsG(mode='N')
        self.mask_block_C = maskBlock()
        self.mask_block_N = maskBlock1()

    def forward(self, rgb, lidar, mask, stage): ###rgb1
        surface_normal ,global_features= self.normal(rgb, lidar, mask ) ##r

        if stage == 'N':
            return None, None, None, None, surface_normal

        color_path_dense, confident_mask, cat2C = self.color_path(rgb, lidar, global_features)###

        normal_path_dense= self.normal_path(surface_normal, lidar, confident_mask)##

        color_attn = self.mask_block_C(cat2C)

        normal_attn = self.mask_block_N(normal_path_dense)

        return color_path_dense, normal_path_dense, color_attn, normal_attn, surface_normal#

