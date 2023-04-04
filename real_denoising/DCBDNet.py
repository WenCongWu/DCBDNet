import torch
import torch.nn as nn
import torch.nn.functional as F


class Up(nn.Module):

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x):
        x2 = self.up(x1)

        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]
        x3 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x3


class FCN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64):
        super(FCN, self).__init__()
        kernel_size = 3

        self.fconv1 = nn.Conv2d(in_nc, nc, kernel_size=kernel_size, padding=1)
        self.frelu1 = nn.ReLU(inplace=True)

        self.fconv2 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=1)
        self.fbn2 = nn.BatchNorm2d(nc)
        self.frelu2 = nn.ReLU(inplace=True)

        self.mp = nn.MaxPool2d(2)

        self.fconv3 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=1)
        self.fbn3 = nn.BatchNorm2d(nc)
        self.frelu3 = nn.ReLU(inplace=True)

        self.fconv4 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=1)
        self.fbn4 = nn.BatchNorm2d(nc)
        self.frelu4 = nn.ReLU(inplace=True)

        self.fconv5 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=1)
        self.fbn5 = nn.BatchNorm2d(nc)
        self.frelu5 = nn.ReLU(inplace=True)

        self.fconv6 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=1)
        self.fbn6 = nn.BatchNorm2d(nc)
        self.frelu6 = nn.ReLU(inplace=True)

        self.fconv7 = nn.Conv2d(nc, out_nc, kernel_size=kernel_size, padding=1)

        self.tanh_mapping = nn.Tanh()

        self.up = Up()

    def forward(self, x):
        x1 = self.fconv1(x)
        x1 = self.frelu1(x1)

        x2 = self.fconv2(x1)
        x2 = self.fbn2(x2)
        x2 = self.frelu2(x2)

        x3 = self.mp(x2)

        x4 = self.fconv3(x3)
        x4 = self.fbn3(x4)
        x4 = self.frelu3(x4)

        x5 = self.fconv4(x4)
        x5 = self.fbn4(x5)
        x5 = self.frelu4(x5)

        x6 = self.fconv5(x5)
        x6 = self.fbn5(x6)
        x6 = self.frelu5(x6)

        x7 = self.up(x6, x2)

        x8 = self.fconv6(x7 + x2)
        x8 = self.fbn6(x8)
        x8 = self.frelu6(x8)

        x9 = self.fconv7(x8)

        noise_level = self.tanh_mapping(x9)
        return noise_level


class DCBDNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=3, nc=64, bias=True):
        super(DCBDNet, self).__init__()
        kernel_size = 3
        self.fcn = FCN(in_nc=in_nc, out_nc=out_nc, nc=nc)

        self.conv1 = nn.Conv2d(2 * in_nc, nc, kernel_size=kernel_size, bias=bias, padding=1)
        self.bn1 = nn.BatchNorm2d(nc)
        self.relu0 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=1)
        self.bn2 = nn.BatchNorm2d(nc)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=1)
        self.bn3 = nn.BatchNorm2d(nc)
        self.relu2 = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=1)
        self.bn4 = nn.BatchNorm2d(nc)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=1)
        self.bn5 = nn.BatchNorm2d(nc)
        self.relu4 = nn.ReLU(inplace=True)

        self.mp2 = nn.MaxPool2d(2)

        self.conv6 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=1)
        self.bn6 = nn.BatchNorm2d(nc)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=1)
        self.bn7 = nn.BatchNorm2d(nc)
        self.relu6 = nn.ReLU(inplace=True)

        # upsample ignore
        self.conv8 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=1)
        self.bn8 = nn.BatchNorm2d(nc)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv9 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=1)
        self.bn9 = nn.BatchNorm2d(nc)
        self.relu8 = nn.ReLU(inplace=True)

        # upsample ignore
        self.conv10 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=1)
        self.bn10 = nn.BatchNorm2d(nc)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv11 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=1)
        self.bn11 = nn.BatchNorm2d(nc)
        self.relu10 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=1)

        self.m_dilatedconv1 = nn.Conv2d(2 * in_nc, nc, kernel_size=kernel_size, bias=bias, padding=1, dilation=1)
        self.m_bn1 = nn.BatchNorm2d(nc)
        self.m_relu0 = nn.ReLU(inplace=True)

        self.m_dilatedconv2 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=2, dilation=2)
        self.m_bn2 = nn.BatchNorm2d(nc)
        self.m_relu1 = nn.ReLU(inplace=True)

        self.m_dilatedconv3 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=3, dilation=3)
        self.m_bn3 = nn.BatchNorm2d(nc)
        self.m_relu2 = nn.ReLU(inplace=True)

        self.m_dilatedconv4 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=4, dilation=4)
        self.m_bn4 = nn.BatchNorm2d(nc)
        self.m_relu3 = nn.ReLU(inplace=True)

        self.m_dilatedconv5 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=5, dilation=5)
        self.m_bn5 = nn.BatchNorm2d(nc)
        self.m_relu4 = nn.ReLU(inplace=True)

        self.m_dilatedconv6 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=6, dilation=6)
        self.m_bn6 = nn.BatchNorm2d(nc)
        self.m_relu5 = nn.ReLU(inplace=True)

        self.m_dilatedconv5_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=5, dilation=5)
        self.m_bn5_1 = nn.BatchNorm2d(nc)
        self.m_relu4_1 = nn.ReLU(inplace=True)

        self.m_dilatedconv4_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=4, dilation=4)
        self.m_bn4_1 = nn.BatchNorm2d(nc)
        self.m_relu3_1 = nn.ReLU(inplace=True)

        self.m_dilatedconv3_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=3, dilation=3)
        self.m_bn3_1 = nn.BatchNorm2d(nc)
        self.m_relu2_1 = nn.ReLU(inplace=True)

        self.m_dilatedconv2_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=2, dilation=2)
        self.m_bn2_1 = nn.BatchNorm2d(nc)
        self.m_relu1_1 = nn.ReLU(inplace=True)

        self.m_dilatedconv1_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=1, dilation=1)
        self.m_bn1_1 = nn.BatchNorm2d(nc)
        self.m_relu1_1 = nn.ReLU(inplace=True)

        self.m_conv = nn.Conv2d(nc, nc, kernel_size=kernel_size, bias=bias, padding=1, dilation=1)

        self.m_conv_tail = nn.Conv2d(2 * nc, out_nc, kernel_size=kernel_size, bias=bias, padding=1)

        self.up = Up()

    def forward(self, x0):
        noise_level = self.fcn(x0)
        x0 = torch.cat((x0, noise_level), dim=1)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = self.relu0(x1)
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu1(x2)
        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu2(x3)

        x3_1 = self.mp1(x3)

        x4 = self.conv4(x3_1)
        x4 = self.bn4(x4)
        x4 = self.relu3(x4)
        x5 = self.conv5(x4)
        x5 = self.bn5(x5)
        x5 = self.relu4(x5)

        x5_1 = self.mp2(x5)

        x6 = self.conv6(x5_1)
        x6 = self.bn6(x6)
        x6 = self.relu5(x6)
        x7 = self.conv7(x6)
        x7 = self.bn7(x7)
        x7 = self.relu6(x7)

        x7_1 = self.up(x7, x5)

        x8 = self.conv8(x7_1 + x5)
        x8 = self.bn8(x8)
        x8 = self.relu7(x8)
        x9 = self.conv9(x8)
        x9 = self.bn9(x9)
        x9 = self.relu8(x9)

        x9_1 = self.up(x9, x3)

        x10 = self.conv10(x9_1 + x3)
        x10 = self.bn10(x10)
        x10 = self.relu9(x10)
        x11 = self.conv11(x10)
        x11 = self.bn11(x11)
        x11 = self.relu10(x11)
        x = self.conv12(x11)

        y1 = self.m_dilatedconv1(x0)
        y1_1 = self.m_bn1(y1)
        y1_1 = self.m_relu0(y1_1)

        y2 = self.m_dilatedconv2(y1_1)
        y2_1 = self.m_bn2(y2)
        y2_1 = self.m_relu1(y2_1)

        y3 = self.m_dilatedconv3(y2_1)
        y3_1 = self.m_bn3(y3)
        y3_1 = self.m_relu2(y3_1)

        y4 = self.m_dilatedconv4(y3_1)
        y4_1 = self.m_bn4(y4)
        y4_1 = self.m_relu3(y4_1)

        y5 = self.m_dilatedconv5(y4_1)
        y5_1 = self.m_bn5(y5)
        y5_1 = self.m_relu4(y5_1)

        y6 = self.m_dilatedconv6(y5_1)
        y6_1 = self.m_bn6(y6)
        y6_1 = self.m_relu5(y6_1)

        y7 = self.m_dilatedconv5_1(y6_1)
        y7 = self.m_bn5_1(y7)
        y7 = self.m_relu4_1(y7)

        y8 = self.m_dilatedconv4_1(y7 + y5)
        y8 = self.m_bn4_1(y8)
        y8 = self.m_relu3_1(y8)

        y9 = self.m_dilatedconv3_1(y8 + y4)
        y9 = self.m_bn3_1(y9)
        y9 = self.m_relu2_1(y9)

        y10 = self.m_dilatedconv2_1(y9 + y3)
        y10 = self.m_bn2_1(y10)
        y10 = self.m_relu1_1(y10)

        y11 = self.m_dilatedconv1_1(y10 + y2)
        y11 = self.m_bn1_1(y11)
        y11 = self.m_relu1_1(y11)

        y = self.m_conv(y11 + y1)

        z = torch.cat([x, y], dim=1)
        Z = self.m_conv_tail(z)
        return Z, noise_level
