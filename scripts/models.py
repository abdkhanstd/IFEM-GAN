import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import spectral_norm


class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(DynamicConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dynamic_weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dynamic_weight, a=math.sqrt(5))

    def forward(self, x):
        batch_size = x.size(0)
        weight = self.dynamic_weight.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        x = x.view(1, batch_size * self.in_channels, x.size(2), x.size(3))
        weight = weight.view(batch_size * self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        output = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_channels, output.size(2), output.size(3))
        return output

class SPEM(nn.Module):
    def __init__(self, in_channels=128, kernel_size=7):
        super(SPEM, self).__init__()
        self.layers = nn.Sequential(
            DynamicConv2d(in_channels, 32, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DynamicConv2d(32, 64, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            DynamicConv2d(64, 128, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            DynamicConv2d(128, 128, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            DynamicConv2d(128, 64, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            DynamicConv2d(64, 32, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

from torch_dct import dct, idct

class SpectralAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SpectralAttention, self).__init__()
        self.in_planes = in_planes
        self.fc1 = nn.Linear(in_planes, in_planes // ratio, bias=False)
        self.fc2 = nn.Linear(in_planes // ratio, in_planes, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc1(x)
        avg_out = F.relu(avg_out)
        avg_out = self.fc2(avg_out)
        return self.sigmoid(avg_out)

class SFOM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SFOM, self).__init__()
        self.in_planes = in_planes
        self.fc1 = nn.Linear(in_planes, in_planes // ratio, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_planes // ratio, in_planes, bias=False)
        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Spectral Attention Mechanism
        self.spectral_attention = SpectralAttention(in_planes, ratio)

    def forward(self, x):
        # Apply DCT
        batch_size, channels, height, width = x.size()
        dct_x = dct(x.view(batch_size, channels, -1), norm='ortho').view(batch_size, channels, height, width)
        

        # Apply frequency-based gating
        frequency_gating = self.spectral_attention(dct_x.mean(dim=[2, 3]))
        gated_dct_x = dct_x * frequency_gating.unsqueeze(2).unsqueeze(3)

        # Apply inverse DCT
        idct_x = idct(gated_dct_x.view(batch_size, channels, -1), norm='ortho').view(batch_size, channels, height, width)

        # Combine DCT and inverse DCT outputs
        out = (x + idct_x) / 2

        return self.sigmoid(out)


class ConditionalNormalization(nn.Module):
    def __init__(self, in_features):
        super(ConditionalNormalization, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(in_features)
        self.batch_norm = nn.BatchNorm2d(in_features)

    def forward(self, x):
        if self.training or (x.size(2) > 1 and x.size(3) > 1):
            return self.instance_norm(x)
        else:
            return self.batch_norm(x)

class CBAM(nn.Module):
    def __init__(self, in_features):
        super(CBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, 3, padding=1)
        self.norm1 = ConditionalNormalization(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_features, in_features, 3, padding=1)
        self.norm2 = ConditionalNormalization(in_features)
        self.ca = SFOM(in_features)
        self.sa = SPEM(in_features)

    def forward(self, x):
        if x.size(2) > 1 and x.size(3) > 1:
            x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        out = self.conv1(x)
        if out.size(2) > 1 and out.size(3) > 1:
            out = self.norm1(out)
        out = self.relu(out)
        if out.size(2) > 1 and out.size(3) > 1:
            out = F.pad(out, (1, 1, 1, 1), mode='reflect')
        out = self.conv2(out)
        if out.size(2) > 1 and out.size(3) > 1:
            out = self.norm2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        return out


class GPSM(nn.Module):
    def __init__(self, in_features):
        super(GPSM, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
        self.SFOM_layer = SFOM(in_features)
        self.CBAM_layer = CBAM(in_features)

    def forward(self, x):
        identity = x
        out = self.conv_block(x)
        out = self.SFOM_layer(out)
        out = self.CBAM_layer(out)
        if out.shape != identity.shape:
            out = F.interpolate(out, size=identity.shape[2:])
        out += identity
        return F.relu(out)

class TrainableSmoothingLayer(nn.Module):
    def __init__(self, in_channels):
        super(TrainableSmoothingLayer, self).__init__()
        initial_kernel = torch.tensor([[1.0, 1.0, 1.0],
                                       [1.0, 1.0, 1.0],
                                       [1.0, 1.0, 1.0]]) / 9.0
        self.kernel = nn.Parameter(initial_kernel.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1))
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)

    def forward(self, x):
        self.conv.weight.data = self.kernel
        return self.conv(x)

class TrainableSharpeningLayer(nn.Module):
    def __init__(self, in_channels):
        super(TrainableSharpeningLayer, self).__init__()
        initial_kernel = torch.tensor([[ 0.0,-1.0, 0.0],
                                       [-1.0, 5.0,-1.0],
                                       [ 0.0,-1.0, 0.0]])
        self.kernel = nn.Parameter(initial_kernel.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1))
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)

    def forward(self, x):
        self.conv.weight.data = self.kernel
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_GPSM=2):
        super(Generator, self).__init__()
        in_features = 64

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, in_features, 7),
                    nn.InstanceNorm2d(in_features),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        depth = 2
        for _ in range(depth):
            out_features = in_features * 2
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features

        # Residual blocks
        for _ in range(n_GPSM):
            model += [GPSM(in_features)]

        # Upsampling
        for _ in range(depth):
            out_features = in_features // 2
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)
        self.smoothing = TrainableSmoothingLayer(3)
        self.sharpening = TrainableSharpeningLayer(3)

    def forward(self, x):
        x = self.model(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            
            nn.Conv2d(256, 512, kernel_size=4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )
        
        # Apply spectral normalization to the convolutional layers
        self.apply_spectral_norm()

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    
    def apply_spectral_norm(self):
        for layer in self.model:
            if isinstance(layer, nn.Conv2d):
                nn.utils.spectral_norm(layer)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('DynamicConv2d') != -1:
        if hasattr(m, 'dynamic_weight') and m.dynamic_weight is not None:
            torch.nn.init.normal_(m.dynamic_weight.data, 0.0, 0.02)
    elif classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)
