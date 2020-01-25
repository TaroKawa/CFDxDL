from __future__ import print_function

import random

import torch.nn as nn
import torch.nn.functional as Fun
import torch.utils.data
import torch.utils.data

#from shape_to_shock.parameter import parameters
from parameter import parameters

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

temp = parameters()
parameters = temp.__dict__


######################################################################
# Encoder
# Encoder Code
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.channel = parameters["input_channel"]
        self.conv_channel1 = parameters["conv_channel1"]
        self.conv_channel2 = parameters["conv_channel2"]
        self.conv_channel3 = parameters["conv_channel3"]
        self.conv_channel4 = parameters["conv_channel4"]
        self.conv_channel5 = parameters["conv_channel5"]

        self.conv_kernel1 = parameters["conv_kernel_1"]
        self.conv_kernel2 = parameters["conv_kernel_2"]
        self.conv_kernel3 = parameters["conv_kernel_3"]
        self.conv_kernel4 = parameters["conv_kernel_4"]
        self.conv_kernel5 = parameters["conv_kernel_5"]

        self.conv_stride1 = parameters["conv_stride_1"]
        self.conv_stride2 = parameters["conv_stride_2"]
        self.conv_stride3 = parameters["conv_stride_3"]
        self.conv_stride4 = parameters["conv_stride_4"]
        self.conv_stride5 = parameters["conv_stride_5"]

        self.model = nn.Sequential(
            # layer 1
            # nn.Conv2d(self.channel, self.conv_channel1, kernel_size=4, stride=4, bias=True),
            nn.Conv2d(self.channel, self.conv_channel1, kernel_size=self.conv_kernel1, stride=self.conv_stride1,
                      bias=True),
            nn.BatchNorm2d(self.conv_channel1),
            nn.LeakyReLU(0.1),

            # layer 2
            # nn.Conv2d(self.conv_channel1, self.conv_channel2, kernel_size=4, stride=4, bias=True),
            nn.Conv2d(self.conv_channel1, self.conv_channel2, kernel_size=self.conv_kernel2, stride=self.conv_stride2,
                      bias=True),
            nn.BatchNorm2d(self.conv_channel2),
            nn.LeakyReLU(0.1),

            # layer 3
            # nn.Conv2d(self.conv_channel2, self.conv_channel3, kernel_size=4, stride=4, bias=True),
            nn.Conv2d(self.conv_channel2, self.conv_channel3, kernel_size=self.conv_kernel3, stride=self.conv_stride3,
                      bias=True),
            nn.BatchNorm2d(self.conv_channel3),
            nn.LeakyReLU(0.1),

            # layer 4
            # nn.Conv2d(self.conv_channel2, self.conv_channel3, kernel_size=4, stride=4, bias=True),
            nn.Conv2d(self.conv_channel3, self.conv_channel4, kernel_size=self.conv_kernel4, stride=self.conv_stride4,
                      bias=True),
            nn.BatchNorm2d(self.conv_channel4),
            nn.LeakyReLU(0.1),

            # layer 5
            # nn.Conv2d(self.conv_channel2, self.conv_channel3, kernel_size=4, stride=4, bias=True),
            nn.Conv2d(self.conv_channel4, self.conv_channel5, kernel_size=self.conv_kernel5, stride=self.conv_stride5,
                      bias=True),
            nn.BatchNorm2d(self.conv_channel5),
            nn.LeakyReLU(0.1),

        )

    def forward(self, input):
        return self.model(input)


######################################################################
# concate
# ~~~~~~~~~
# Linear Code

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.n_hidden = parameters["n_hidden"]
        self.input_channel = parameters["conv_channel5"]
        self.size = parameters["linear_size"]

        self.fc1 = nn.Linear(self.input_channel * self.size * self.size, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden + 2, self.input_channel * self.size * self.size)

    #        self.model = nn.Sequential(
    #            # layer 1
    #            nn.Linear(self.conv_channel3 * self.size ** 2 + 2, self.n_hidden),
    #            nn.LeakyReLU(0.1),
    #            nn.Linear(self.n_hidden, self.conv_channel3 * self.size ** 2)
    #        )
    #
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x, input_labels):
        #print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        #x = nn.LeakyReLU(0.1)(self.fc1(x))
        x = nn.Sigmoid()(self.fc1(x))
        #print(input_labels)
        x = torch.cat((x, input_labels), 1)
        #print(self.size)
        x = self.fc2(x)
        return torch.reshape(x, (-1, self.input_channel, self.size, self.size))


######################################################################
# Decoder
# ~~~~~~~~~
# Decoder Code
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.output_channel = parameters["output_channel"]
        self.conv_channel1 = parameters["conv_channel1"]
        self.conv_channel2 = parameters["conv_channel2"]
        self.conv_channel3 = parameters["conv_channel3"]
        self.conv_channel4 = parameters["conv_channel4"]
        self.conv_channel5 = parameters["conv_channel5"]

        self.conv_kernel1 = parameters["conv_kernel_1"]
        self.conv_kernel2 = parameters["conv_kernel_2"]
        self.conv_kernel3 = parameters["conv_kernel_3"]
        self.conv_kernel4 = parameters["conv_kernel_4"]
        self.conv_kernel5 = parameters["conv_kernel_5"]

        self.conv_stride1 = parameters["conv_stride_1"]
        self.conv_stride2 = parameters["conv_stride_2"]
        self.conv_stride3 = parameters["conv_stride_3"]
        self.conv_stride4 = parameters["conv_stride_4"]
        self.conv_stride5 = parameters["conv_stride_5"]

        self.model = nn.Sequential(
            # layer 1
            # nn.ConvTranspose2d(self.conv_channel3, self.conv_channel2, kernel_size=4, stride=4, bias=True),
            # nn.ConvTranspose2d(self.conv_channel3, self.conv_channel2, kernel_size=3, stride=1, bias=True),
            nn.ConvTranspose2d(self.conv_channel5, self.conv_channel4, kernel_size=self.conv_kernel5,
                               stride=self.conv_stride5, bias=True),
            nn.BatchNorm2d(self.conv_channel4),
            nn.LeakyReLU(0.1),

            # layer 2
            # nn.ConvTranspose2d(self.conv_channel2, self.conv_channel1, kernel_size=6, stride=4, bias=True),
            # nn.ConvTranspose2d(self.conv_channel2, self.conv_channel1, kernel_size=3, stride=2, bias=True),
            nn.ConvTranspose2d(self.conv_channel4, self.conv_channel3, kernel_size=self.conv_kernel4,
                               stride=self.conv_stride4, bias=True),
            nn.BatchNorm2d(self.conv_channel3),
            nn.LeakyReLU(0.1),
            #

            # layer 3
            # nn.ConvTranspose2d(self.conv_channel1, 50, kernel_size=3, stride=1, bias=True),
            # nn.ConvTranspose2d(self.conv_channel3, self.conv_channel2, kernel_size=4, stride=1, bias=True),
            nn.ConvTranspose2d(self.conv_channel3, self.conv_channel2, kernel_size=self.conv_kernel3,
                               stride=self.conv_stride3, bias=True),
            nn.BatchNorm2d(self.conv_channel2),
            nn.LeakyReLU(0.1),

            # layer 4
            # nn.ConvTranspose2d(self.conv_channel1, 50, kernel_size=3, stride=1, bias=True),
            # nn.ConvTranspose2d(self.conv_channel2, self.conv_channel1, kernel_size=4, stride=1, bias=True),
            nn.ConvTranspose2d(self.conv_channel2, self.conv_channel1, kernel_size=self.conv_kernel2,
                               stride=self.conv_stride2, bias=True),
            nn.BatchNorm2d(self.conv_channel1),
            nn.LeakyReLU(0.1),

            # layer 5
            # nn.ConvTranspose2d(self.conv_channel1, 50, kernel_size=3, stride=1, bias=True),
            # nn.ConvTranspose2d(self.conv_channel1, self.conv_chann, kernel_size=4, stride=1, bias=True),
            #nn.ConvTranspose2d(self.conv_channel1, 10, kernel_size=self.conv_kernel1, stride=self.conv_stride1,
            #                   bias=True),
            nn.ConvTranspose2d(self.conv_channel1, self.output_channel, kernel_size=self.conv_kernel1, stride=self.conv_stride1,
                               bias=True),

            #nn.BatchNorm2d(10),
            #nn.LeakyReLU(0.1),

            #nn.Conv2d(10, 10, kernel_size=1, stride=1, bias=True),
            #nn.Conv2d(10, 10, kernel_size=1, stride=1, bias=True),
            #nn.Conv2d(10, 10, kernel_size=1, stride=1, bias=True),
            #nn.Conv2d(10, 10, kernel_size=1, stride=1, bias=True),
            #nn.Conv2d(10, self.output_channel, kernel_size=1, stride=1, bias=True),
            #nn.LeakyReLU(0.1),

            # nn.Conv2d(self.output_channel, self.output_channel, kernel_size=1, stride=1, bias=True),
            # nn.Conv2d(self.output_channel, self.output_channel, kernel_size=1, stride=1, bias=True),

            # nn.Sigmoid(),
            # nn.Conv2d(self.output_channel, self.output_channel, kernel_size=1, stride=1, bias=True),
            # nn.LeakyReLU(0.5)
        )

    def forward(self, input):
        return self.model(input)


"""# Unet"""


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            #nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = Fun.pad(x1, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.outconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        #    nn.Sigmoid()
            nn.LeakyReLU(0.1),
        )
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # return self.conv(x)
        return torch.clamp(self.outconv(x), min=0.0, max=1.0)


class Conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1),

        #    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        #    nn.BatchNorm2d(out_channels),
        #    #nn.ReLU(inplace=True),
        #    nn.LeakyReLU(0.1),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=parameters["output_channel"], n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.seed_channel = parameters["unet_seed_channel"]

        self.conv = Conv(self.n_channels, self.n_classes)

        self.inc = DoubleConv(n_channels, self.seed_channel)
        self.down1 = Down(self.seed_channel, self.seed_channel * 2)
        self.down2 = Down(self.seed_channel * 2, self.seed_channel * 4)
        self.down3 = Down(self.seed_channel * 4, self.seed_channel * 8)
        self.down4 = Down(self.seed_channel * 8, self.seed_channel * 8)
        self.up1 = Up(self.seed_channel * 16, self.seed_channel * 4, bilinear)
        self.up2 = Up(self.seed_channel * 8, self.seed_channel * 2, bilinear)
        self.up3 = Up(self.seed_channel * 4, self.seed_channel, bilinear)
        self.up4 = Up(self.seed_channel * 2, self.seed_channel, bilinear)
        self.outc = OutConv(self.seed_channel, n_classes)
        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, bilinear)
        # self.up2 = Up(512, 128, bilinear)
        # self.up3 = Up(256, 64, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)

    def forward(self, x):
        #shock_prediction = self.conv(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        shock_prediction = self.outc(x)
        return shock_prediction


"""# ShockNet"""


######################################################################
# ShockNet
class ShockNet(nn.Module):
    def __init__(self):
        super(ShockNet, self).__init__()
        self.encoder = Encoder()
        self.linear = Linear()
        self.decoder = Decoder()
        self.unet = UNet()

    def _forward(self, distance, input_labels):
        # Encoder
        x = self.encoder(distance)

        x = self.linear(x, input_labels)

        # Decoder
        x = self.decoder(x)
        fluid = x
        # print(x.shape)

        # Shock Detection
        labels = self.unet(x)

        return fluid, labels
        # return fluid

    def forward(self, distance, input_labels):
        x = self._forward(distance, input_labels)
        return x
