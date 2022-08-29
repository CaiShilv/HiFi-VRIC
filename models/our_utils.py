import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from compressai.layers import *
from .layers import IAT, AttentionBlock


class AttModule(nn.Module):
    def __init__(self, N):
        super(AttModule, self).__init__()
        self.forw_att = AttentionBlock(N)
        self.back_att = AttentionBlock(N)

    def forward(self, x, rev=False):
        if not rev:
            return self.forw_att(x)
        else:
            return self.back_att(x)



class EnhModule(nn.Module):
    def __init__(self, nf):
        super(EnhModule, self).__init__()
        self.forw_enh = EnhBlock(nf)
        self.back_enh = EnhBlock(nf)

    def forward(self, x, rev=False):
        if not rev:
            return self.forw_enh(x)
        else:
            return self.back_enh(x)

class EnhBlock(nn.Module):
    def __init__(self, nf):
        super(EnhBlock, self).__init__()
        self.layers = nn.Sequential(
            DenseBlock(3, nf),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            DenseBlock(nf, 3)
        )

    def forward(self, x):
        return x + self.layers(x) * 0.2


class InvComp(nn.Module):
    def __init__(self, N, M):
        super(InvComp, self).__init__()
        self.in_nc = N
        self.out_nc = M
        self.operations1 = nn.ModuleList()
        self.operations2 = nn.ModuleList()
        self.operations3 = nn.ModuleList()
        self.operations4 = nn.ModuleList()

        # 1st level
        b1 = SqueezeLayer(2)
        self.operations1.append(b1)
        self.in_nc *= 4
        b1 = InvertibleConv1x1(self.in_nc)
        self.operations1.append(b1)
        b1 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
        self.operations1.append(b1)
        b1 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
        self.operations1.append(b1)
        b1 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
        self.operations1.append(b1)
        self.SFTlevel_1 = IAT(x_nc=self.in_nc, prior_nc=1, ks=3, nhidden=self.in_nc*2)


        # 2nd level
        b2 = SqueezeLayer(2)
        self.operations2.append(b2)
        self.in_nc *= 4
        b2 = InvertibleConv1x1(self.in_nc)
        self.operations2.append(b2)
        b2 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
        self.operations2.append(b2)
        b2 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
        self.operations2.append(b2)
        b2 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
        self.operations2.append(b2)
        self.SFTlevel_2 = IAT(x_nc=self.in_nc, prior_nc=1, ks=3, nhidden=self.in_nc*2)


        # 3rd level
        b3 = SqueezeLayer(2)
        self.operations3.append(b3)
        self.in_nc *= 4
        b3 = InvertibleConv1x1(self.in_nc)
        self.operations3.append(b3)
        b3 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
        self.operations3.append(b3)
        b3 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
        self.operations3.append(b3)
        b3 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
        self.operations3.append(b3)
        self.SFTlevel_3 = IAT(x_nc=self.in_nc, prior_nc=1, ks=3, nhidden=self.in_nc*2)

        # 4th level
        b4 = SqueezeLayer(2)
        self.operations4.append(b4)
        self.in_nc *= 4
        b4 = InvertibleConv1x1(self.in_nc)
        self.operations4.append(b4)
        b4 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
        self.operations4.append(b4)
        b4 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
        self.operations4.append(b4)
        b4 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
        self.operations4.append(b4)
        self.SFTlevel_4 = IAT(x_nc=self.in_nc//4, prior_nc=1, ks=3, nhidden=(self.in_nc//4)*2)

    def forward(self, x, qmap,rev=False):
        if not rev:
            for op in self.operations1:
                x = op.forward(x, False)
            # qmap1 = F.adaptive_avg_pool2d(qmap, x.size()[2:])
            x = self.SFTlevel_1(x, qmap, reverse=False)

            for op in self.operations2:
                x = op.forward(x, False)
            # qmap2 = F.adaptive_avg_pool2d(qmap, x.size()[2:])
            x = self.SFTlevel_2(x, qmap, reverse=False)

            for op in self.operations3:
                x = op.forward(x, False)
            # qmap3 = F.adaptive_avg_pool2d(qmap, x.size()[2:])
            x = self.SFTlevel_3(x, qmap, reverse=False)

            for op in self.operations4:
                x = op.forward(x, False)
            # qmap4 = F.adaptive_avg_pool2d(qmap, x.size()[2:])

            b, c, h, w = x.size()
            x = torch.mean(x.view(b, c // self.out_nc, self.out_nc, h, w), dim=1)

            x = self.SFTlevel_4(x, qmap, reverse=False)
        else:
            # qmap4 = F.adaptive_avg_pool2d(qmap, x.size()[2:])
            x = self.SFTlevel_4(x, qmap, reverse=True)
            times = self.in_nc // self.out_nc
            x = x.repeat(1, times, 1, 1)


            for op in reversed(self.operations4):
                x = op.forward(x, True)

            # qmap3 = F.adaptive_avg_pool2d(qmap, x.size()[2:])
            x = self.SFTlevel_3(x, qmap, reverse=True)
            for op in reversed(self.operations3):
                x = op.forward(x, True)

            # qmap2 = F.adaptive_avg_pool2d(qmap, x.size()[2:])
            x = self.SFTlevel_2(x, qmap, reverse=True)
            for op in reversed(self.operations2):
                x = op.forward(x, True)

            # qmap1 = F.adaptive_avg_pool2d(qmap, x.size()[2:])
            x = self.SFTlevel_1(x, qmap, reverse=True)
            for op in reversed(self.operations1):
                x = op.forward(x, True)

        return x


class CouplingLayer(nn.Module):
    def __init__(self, split_len1, split_len2, kernal_size, clamp=1.0):
        super(CouplingLayer, self).__init__()
        self.split_len1 = split_len1
        self.split_len2 = split_len2
        self.clamp = clamp

        self.G1 = Bottleneck(self.split_len1, self.split_len2, kernal_size)
        self.G2 = Bottleneck(self.split_len2, self.split_len1, kernal_size)
        self.H1 = Bottleneck(self.split_len1, self.split_len2, kernal_size)
        self.H2 = Bottleneck(self.split_len2, self.split_len1, kernal_size)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            y1 = x1.mul(torch.exp( self.clamp * (torch.sigmoid(self.G2(x2)) * 2 - 1) )) + self.H2(x2)
            y2 = x2.mul(torch.exp( self.clamp * (torch.sigmoid(self.G1(y1)) * 2 - 1) )) + self.H1(y1)
        else:
            y2 = (x2 - self.H1(x1)).div(torch.exp( self.clamp * (torch.sigmoid(self.G1(x1)) * 2 - 1) ))
            y1 = (x1 - self.H2(y2)).div(torch.exp( self.clamp * (torch.sigmoid(self.G2(y2)) * 2 - 1) ))
        return torch.cat((y1, y2), 1)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Bottleneck, self).__init__()
        # P = ((S-1)*W-S+F)/2, with F = filter size, S = stride
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights_xavier([self.conv1, self.conv2], 0.1)
        initialize_weights(self.conv3, 0)
        
    def forward(self, x):
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.lrelu(self.conv2(conv1))
        conv3 = self.conv3(conv2)
        return conv3

class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, reverse=False):
        if not reverse:
            output = self.squeeze2d(input, self.factor)  # Squeeze in forward
            return output
        else:
            output = self.unsqueeze2d(input, self.factor)
            return output
        
    def jacobian(self, x, rev=False):
        return 0
        
    @staticmethod
    def squeeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(B, factor * factor * C, H // factor, W // factor)
        return x

    @staticmethod
    def unsqueeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        factor2 = factor ** 2
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert C % (factor2) == 0, "{}".format(C)
        x = input.view(B, factor, factor, C // factor2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)
        return x

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape

    def get_weight(self, reverse):
        w_shape = self.w_shape
        if not reverse:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            weight = torch.inverse(self.weight.double()).float() \
                .view(w_shape[0], w_shape[1], 1, 1)
        return weight

    def forward(self, input, reverse=False):
        weight = self.get_weight(reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            return z
        else:
            z = F.conv2d(input, weight)
            return z

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
