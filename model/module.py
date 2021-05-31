# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F



# from torchvision import ops
#
# ops.DeformConv2d(input,offest)


class ResBlock1(nn.Module):

    def __init__(self, num_filters, channels_in=None,stride=1,cutnum=16,  numdata=256,res_option='A', use_dropout=False):
        super(ResBlock1, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None

        self.use_dropout = use_dropout

        self.conv = nn.Conv2d(channels_in, 512, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(512, num_filters//2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels_in+num_filters//2, num_filters, kernel_size=3, stride=stride, padding=1 )
        self.bn2 = nn.BatchNorm2d(channels_in+num_filters//2)
        self.relu2 = nn.ReLU(inplace=True)

        # self.conv3 = nn.Conv2d( num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(num_filters)
        # self.relu3 = nn.ReLU(inplace=True)


        if self.use_dropout:
            self.dropout = nn.Dropout(inplace=True)




    def forward(self, x):


        out = self.bn(x)
        out=self.relu(out)
        out=self.conv(out)

        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv1(out)
        if self.use_dropout:
            out = self.dropout(out)

        out1=torch.cat((x,out),1)

        out = self.bn2(out1)
        out = self.relu2(out)
        out = self.conv2(out)


        # if self.use_dropout:
        #     out = self.dropout(out)
        # out = self.bn3(out)
        # out = self.relu3(out)
        # out = self.conv3(out)
        return out


class ResBlock2(nn.Module):

    def __init__(self, num_filters, channels_in=None, stride=1, res_option='A', use_dropout=False):
        super(ResBlock2, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None
        # else:
        #    if res_option == 'A':
        #        self.projection = IdentityPadding(num_filters, channels_in, stride)
        #    elif res_option == 'B':
        #        self.projection = ConvProjection(num_filters, channels_in, stride)
        #    elif res_option == 'C':
        #        self.projection = AvgPoolPadding(num_filters, channels_in, stride)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        if self.use_dropout:
            self.dropout = nn.Dropout(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)

        self.bn3 = nn.BatchNorm2d(num_filters*2)
        self.conv4 = nn.Conv2d(num_filters*2, num_filters, kernel_size=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv5= nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_filters)
        self.relu5 = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)


        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        residual = self.conv3(residual)

        # if self.use_dropout:
        #     out = self.dropout(out)

        out = torch.cat((residual,out),1)

        out=self.bn3(out)
        out = self.relu3(out)
        out = self.conv4(out)

        if self.use_dropout:
            out = self.dropout(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out = self.conv5(out)





        return out


class ResBlock3(nn.Module):

    def __init__(self, num_filters, channels_in=None, stride=1, res_option='A', use_dropout=False):
        super(ResBlock3, self).__init__()

        self.use_dropout = use_dropout


        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None

        if self.use_dropout:
            self.dropout = nn.Dropout(inplace=True)

        self.bn = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(channels_in, 64, kernel_size=1, stride=1, bias=False)

        ##################densenet
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels_in, 32 *4, kernel_size=1, stride=1 ,bias=False)
        self.bn2 = nn.BatchNorm2d(32 *4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d( 32 * 4, 32 ,kernel_size=3, stride=1 ,padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(channels_in +32)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(channels_in +32, 32 * 4, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32 * 4)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32 * 4, 32, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn5 = nn.BatchNorm2d(channels_in +32 *2)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(channels_in +32 *2, 32 * 4, kernel_size=1, stride=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32 * 4)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(32 * 4, 32, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn7 = nn.BatchNorm2d(64+32 *3)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv7= nn.Conv2d(64+32 *3, num_filters, kernel_size=3, stride=stride, padding=1, bias=False)




    def forward(self, x):
        residual = x

        residual = self.bn(residual)
        residual = self.relu(residual)
        residual = self.conv(residual)
        # print(residual.size())


        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out1 =self.conv2(out)
        # print(out1.size())

        out = torch.cat((x, out1), 1)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out2 = self.conv4(out)
        # print(out2.size())

        out = torch.cat((x, out1, out2), 1)

        out = self.bn5(out)
        out = self.relu5(out)
        out = self.conv5(out)
        out = self.bn6(out)
        out = self.relu6(out)
        out3 = self.conv6(out)
        # print(out3.size())

        out = torch.cat((residual, out1, out2, out3), 1)



        out = self.bn7(out)
        out = self.relu7(out)
        out = self.conv7(out)

        if self.use_dropout:
            out = self.dropout(out)

        return out


class ResBlock4(nn.Module):

    def __init__(self, num_filters, channels_in=None, stride=1, res_option='A', use_dropout=False):
        super(ResBlock4, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None
        # else:
        #    if res_option == 'A':
        #        self.projection = IdentityPadding(num_filters, channels_in, stride)
        #    elif res_option == 'B':
        #        self.projection = ConvProjection(num_filters, channels_in, stride)
        #    elif res_option == 'C':
        #        self.projection = AvgPoolPadding(num_filters, channels_in, stride)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        if self.use_dropout:
            self.dropout = nn.Dropout(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)

        self.conv4 = nn.Conv2d(num_filters*2, num_filters, kernel_size=1, stride=1)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.bn3(self.conv3(residual))

        if self.use_dropout:
            out = self.dropout(out)


        out += residual

        out=torch.cat((out,residual),1)
        out=self.conv4(out)
        out = self.relu2(out)
        return out

class ResBlock(nn.Module):
    
    def __init__(self, num_filters, channels_in=None, stride=1, res_option='A', use_dropout=False):
        super(ResBlock, self).__init__()
        
        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None
        #else:
        #    if res_option == 'A':
        #        self.projection = IdentityPadding(num_filters, channels_in, stride)
        #    elif res_option == 'B':
        #        self.projection = ConvProjection(num_filters, channels_in, stride)
        #    elif res_option == 'C':
        #        self.projection = AvgPoolPadding(num_filters, channels_in, stride)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

        #################CBAM注意力
        # self.ca = ChannelAttention(num_filters)
        # self.sa = SpatialAttention()


        if self.use_dropout:
            self.dropout = nn.Dropout(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)


        residual = self.bn3(self.conv3(residual))

        if self.use_dropout:
            out = self.dropout(out)

        out += residual
        ################CBAM注意力
        # out = self.ca(out) * out  # 广播机制
        # out = self.sa(out) * out  # 广播机制
        out = self.relu2(out)
        return out


class ResBlock7(nn.Module):

    def __init__(self, num_filters, channels_in=None, stride=1, res_option='A', use_dropout=False):
        super(ResBlock7, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None

        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)



        if self.use_dropout:
            self.dropout = nn.Dropout(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.conv4 = DeformConv2d(num_filters, num_filters, 3, padding=1, stride=1, bias=False, modulation=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv4(out)
        out = self.bn2(out)

        residual = self.bn3(self.conv3(residual))

        if self.use_dropout:
            out = self.dropout(out)

        out += residual
        ################CBAM注意力
        # out = self.ca(out) * out  # 广播机制
        # out = self.sa(out) * out  # 广播机制
        out = self.relu2(out)
        # out=self.conv4(out)
        return out



class ResBlock5(nn.Module):

    def __init__(self, num_filters, channels_in=None, stride=1, res_option='A', use_dropout=True):
        super(ResBlock5, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None
        # else:
        #    if res_option == 'A':
        #        self.projection = IdentityPadding(num_filters, channels_in, stride)
        #    elif res_option == 'B':
        #        self.projection = ConvProjection(num_filters, channels_in, stride)
        #    elif res_option == 'C':
        #        self.projection = AvgPoolPadding(num_filters, channels_in, stride)
        self.use_dropout = use_dropout



        self.conv1 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)



        self.conv2 = DeformConv2d(num_filters,num_filters , 3, padding=1, stride=1,bias=False, modulation=True)
            # nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU(inplace=True)

        # self.conv2 = DeformConv2d(num_filters, num_filters, 3, padding=1, stride=1, bias=False, modulation=True)
        # # nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        # self.bn2 = nn.BatchNorm2d(num_filters)
        # self.relu2 = nn.ReLU(inplace=True)

        # self.conv3 = nn.Conv2d(channels_in, num_filters, kernel_size=3, padding=1,stride=stride,bias=False)
        # self.bn3 = nn.BatchNorm2d(num_filters)
        # self.offsets = nn.Conv2d(num_filters, 18, kernel_size=3, padding=1,stride=1, bias=False)
        # self.conv4 = ops.DeformConv2d(num_filters, num_filters, kernel_size=3, padding=1,stride=1, bias=False)


        # #################CBAM注意力
        # self.ca = ChannelAttention(num_filters)
        # self.sa = SpatialAttention()

        if self.use_dropout:
            self.dropout = nn.Dropout(inplace=True)

        # self.relu1 = nn.ReLU(inplace=True)

        # self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(num_filters)
        # self.relu3 = nn.ReLU(inplace=True)
        #
        # self.conv4 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        # self.bn4 = nn.BatchNorm2d(num_filters)

        # self.conv3 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        # self.bn3 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        # residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.relu2(out)
        # out = self.relu1(out)

        # out = self.conv3(out)
        # out = self.bn3(out)
        # out = self.relu3(out)



        # residual = self.bn4(self.conv4(residual))
        if self.use_dropout:
            out = self.dropout(out)
        # out += residual
        out = self.relu2(out)
        return out




class ResBlock6(nn.Module):

    def __init__(self, num_filters, use_dropout=False):
        super(ResBlock6, self).__init__()
        self.use_dropout = use_dropout
        self.conv2 = DeformConv2d(num_filters,num_filters , 3, padding=1, stride=1,bias=False, modulation=True)

        if self.use_dropout:
            self.dropout = nn.Dropout(inplace=True)
        # self.bn2 = nn.BatchNorm2d(num_filters)
        # self.relu2 = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv2(x)
        # out = self.bn2(out)
        if self.use_dropout:
            out = self.dropout(out)
        # out = self.relu2(out)
        return out


class FilterResponseNormNd(nn.Module):

    def __init__(self, ndim, num_features, eps=1e-6,
                 learnable_eps=False):
        """
        Input Variables:
        ----------------
            ndim: An integer indicating the number of dimensions of the expected input tensor.
            num_features: An integer indicating the number of input feature dimensions.
            eps: A scalar constant or learnable variable.
            learnable_eps: A bool value indicating whether the eps is learnable.
        """
        assert ndim in [3, 4, 5], \
            'FilterResponseNorm only supports 3d, 4d or 5d inputs.'
        super(FilterResponseNormNd, self).__init__()
        shape = (1, num_features) + (1,) * (ndim - 2)
        self.eps = nn.Parameter(torch.ones(*shape) * eps)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.Tensor(*shape))
        self.beta = nn.Parameter(torch.Tensor(*shape))
        self.tau = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    def forward(self, x):
        avg_dims = tuple(range(2, x.dim()))
        nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)


class FilterResponseNorm2d(FilterResponseNormNd):

    def __init__(self,channlin,num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm2d, self).__init__(
            channlin, num_features, eps=eps, learnable_eps=learnable_eps)



class UpProject(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpProject, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3))
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2))
        self.conv1_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3))
        self.conv2_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2))
        self.conv2_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2)

        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        batch_size = x.size(0)
        out1_1 = self.conv1_1(nn.functional.pad(x, (1, 1, 1, 1)))

        out1_2 = self.conv1_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github

        out1_3 = self.conv1_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github

        out1_4 = self.conv1_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

        out2_1 = self.conv2_1(nn.functional.pad(x, (1, 1, 1, 1)))

        out2_2 = self.conv2_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github

        out2_3 = self.conv2_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github

        out2_4 = self.conv2_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

        height = out1_1.size()[2]
        width = out1_1.size()[3]

        out1_1_2 = torch.stack((out1_1, out1_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out1_3_4 = torch.stack((out1_3, out1_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out1_1234 = torch.stack((out1_1_2, out1_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out2_1_2 = torch.stack((out2_1, out2_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out2_3_4 = torch.stack((out2_3, out2_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out2_1234 = torch.stack((out2_1_2, out2_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out1 = self.bn1_1(out1_1234)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn2(out1)

        out2 = self.bn1_2(out2_1234)

        out = out1 + out2
        out = self.relu(out)

        return out


def adaptive_cat(out_conv, out_deconv, out_depth_up):
    out_deconv = out_deconv[:, :, :out_conv.size(2), :out_conv.size(3)]
    out_depth_up = out_depth_up[:, :, :out_conv.size(2), :out_conv.size(3)]
    return torch.cat((out_conv, out_deconv, out_depth_up), 1)











#############################注意力'



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)









class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)   ###输出是18  表示 ＸＹ的偏移量


        nn.init.constant_(self.p_conv.weight, 0)###用值val填充向量。
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)



            ###后续定义了self.conv（最终输出的卷积层，设置输入通道数和输出通道数），
            # self.p_conv（偏置层，学习之前公式(2)中说的偏移量），
            # self.m_conv(权重学习层)。
            # register_backward_hook是为了方便查看这几层学出来的结果，对网络结构无影响。

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)####################3


        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset





#＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃ＥＲＦＮＥＴ


class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput, (3, 3), stride=2, padding=1, bias=True)
        # self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, inpu):
        output = self.conv(inpu)# torch.cat([self.conv(inpu), self.pool(inpu)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)


        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1*dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, inpt):

        output = self.conv3x1_1(inpt)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+inpt)


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.initial_block = DownsamplerBlock(in_channels, 16)
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5): ##5
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):##2
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))


    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, nput):
        output = self.conv(nput)
        output = self.bn(output)
        return F.relu(output)


class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layer1 = UpsamplerBlock(128, 64)
        self.layer2 = non_bottleneck_1d(64, 0, 1)
        self.layer3 = non_bottleneck_1d(64, 0, 1) # 64x64x304

        self.layer4 = UpsamplerBlock(64, 32)
        self.layer5 = non_bottleneck_1d(32, 0, 1)
        self.layer6 = non_bottleneck_1d(32, 0, 1) # 32x128x608

        self.output_conv = nn.ConvTranspose2d(32, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        em2 = output#
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.layer6(output)
        em1 = output#

        output = self.output_conv(output)

        return output,em1,em2
class Decoder1 (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layer1 = UpsamplerBlock(128, 64)
        self.layer2 = non_bottleneck_1d(64, 0, 1)
        self.layer3 = non_bottleneck_1d(64, 0, 1) # 64x64x304

        self.layer4 = UpsamplerBlock(64, 32)
        self.layer5 = non_bottleneck_1d(32, 0, 1)
        self.layer6 = non_bottleneck_1d(32, 0, 1) # 32x128x608

        self.output_conv = nn.ConvTranspose2d(32, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        # em2 = output#
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.layer6(output)
        # em1 = output#

        output = self.output_conv(output)

        return output#,em1,em2

class Net(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):  #use encoder to pass pretrained encoder  先把　RGB3通道并上 激光数据  之后得到GIUDIE 一层 与激光雷达数据集进行相加
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)

    def forward(self, input):
        output = self.encoder(input)
        return self.decoder.forward(output)
class Net2(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):  #use encoder to pass pretrained encoder  先把　RGB3通道并上 激光数据  之后得到GIUDIE 一层 与激光雷达数据集进行相加
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder1(out_channels)

    def forward(self, input):
        output = self.encoder(input)
        return self.decoder.forward(output)

class Net1(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):  #use encoder to pass pretrained encoder  先把　RGB3通道并上 激光数据  之后得到GIUDIE 一层 与激光雷达数据集进行相加
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)

    def forward(self, input):
        output = self.encoder(input)
        return self.decoder.forward(output)



class uncertainty_net(nn.Module):
    def __init__(self):
        super(uncertainty_net, self).__init__()

        # self.convbnrelu = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1, 1),nn.ReLU(inplace=True))
        self.convbnrelu = nn.Sequential(convbn(1, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))
        self.hourglass1 = hourglass_1(32)


        # self.hourglass2 = hourglass_2(32)


        # self.fuse = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1, 1),
        self.fuse = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 3, kernel_size=3, padding=1, stride=1, bias=True))

    def forward(self, input,em1,em2):


        out = self.convbnrelu(input)
        out1= self.hourglass1(out,em1,em2)
        out1 = out1 + out


        # out2 = self.hourglass2(out1,em3,em4)
        # out2 = out2 + out


        out = self.fuse(out1)######################################333
        lidar_out = out


        return lidar_out

class uncertainty_net1(nn.Module):
    def __init__(self):
        super(uncertainty_net1, self).__init__()

        # self.convbnrelu = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1, 1),nn.ReLU(inplace=True))
        self.convbnrelu = nn.Sequential(convbn(1, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))
        self.hourglass1 = hourglass_1(32)


        # self.hourglass2 = hourglass_2(32)


        # self.fuse = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1, 1),
        self.fuse = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),



                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 2, kernel_size=3, padding=1, stride=1, bias=True))

    def forward(self, input,em1,em2):


        out = self.convbnrelu(input)
        out1 = self.hourglass1(out,em1,em2)
        out1 = out1 + out


        # out2 = self.hourglass2(out1,em3,em4)
        # out2 = out2 + out


        out = self.fuse(out1)##########################################
        lidar_out = out


        return lidar_out










#
#
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))
                         # nn.BatchNorm2d(out_planes))


#
# def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
#
#     return nn.Sequential(ResBlock12(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))
#                          # nn.BatchNorm2d(out_planes))

class ResBlock12(nn.Module):

    def __init__(self, channels_in, num_filters, kernel_size, stride, padding, dilation, bias=False):
        super(ResBlock12, self).__init__()

        self.conv1 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        # self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(num_filters)

        # self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        # self.bn3 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        residual = self.conv3(residual)
        out += residual
        # out = self.relu2(out)
        return out


class hourglass_1(nn.Module):
    def __init__(self, channels_in=32):
        super(hourglass_1, self).__init__()

        self.conv1 = nn.Sequential(convbn(channels_in, channels_in, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(channels_in, channels_in, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv3 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=1, pad=1, dilation=1))

        self.conv5 = nn.Sequential(nn.ConvTranspose2d(channels_in*4, channels_in*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.ConvTranspose2d(channels_in*2, channels_in, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in))

    def forward(self, x, em1, em2):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = torch.cat((x, em1), 1)

        x_prime = self.conv3(x)
        x_prime = self.conv4(x_prime)
        x_prime = F.relu(x_prime, inplace=True)
        x_prime = torch.cat((x_prime, em2), 1)

        out = self.conv5(x_prime)
        out = self.conv6(out)

        return out#, x, x_prime


class hourglass_2(nn.Module):
    def __init__(self, channels_in=32):
        super(hourglass_2, self).__init__()

        self.conv1 = nn.Sequential(convbn(channels_in, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(channels_in*2, channels_in*2, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv3 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn(channels_in*2, channels_in*4, kernel_size=3, stride=1, pad=1, dilation=1))

        self.conv5 = nn.Sequential(nn.ConvTranspose2d(channels_in*4, channels_in*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.ConvTranspose2d(channels_in*2, channels_in, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in))

    def forward(self, x, em1, em2):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + em1
        x = F.relu(x, inplace=True)

        x_prime = self.conv3(x)
        x_prime = self.conv4(x_prime)
        x_prime = x_prime + em2
        x_prime = F.relu(x_prime, inplace=True)

        out = self.conv5(x_prime)
        out = self.conv6(out)

        return out






class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self,no_spatial=False):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1/3)*(x_out + x_out11 + x_out21)
        else:
            x_out = (1/2)*(x_out11 + x_out21)
        return x_out
if __name__ == '__main__':
    model=uncertainty_net(34,4)
    print(model)
