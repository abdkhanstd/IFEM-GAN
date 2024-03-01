import torch.nn as nn
import torch.nn.functional as F
import torch
# 想办法加进ResidualBlock里

import cv2
import numpy as np

from skimage.restoration import denoise_tv_chambolle

def apply_fft_filter(image):
    # Convert the image to gray-scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute the 2D FFT
    f = np.fft.fft2(gray)
    # Shift the DC component to the center
    fshift = np.fft.fftshift(f)
    
    # Create a mask here with the appropriate size (for example, a circular mask)
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    # Apply mask to the FFT shifted image
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    
    # Inverse shift to get the original position
    f_ishift = np.fft.ifftshift(fshift)
    # Inverse FFT to get the image back
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return img_back.astype(np.uint8)


## Need to test these two blocks
class CBAMBlock(nn.Module):
    def __init__(self, in_features):
        super(CBAMBlock, self).__init__()
        self.in_features = in_features
        self.conv1 = nn.Conv2d(in_features, in_features, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_features, in_features, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(in_features)
        self.ca = ChannelAttention(in_features)
        self.sa = SpatialAttention()

    def forward(self, x):
        # Apply reflection padding conditionally
        if x.size(2) > 1 and x.size(3) > 1:
            x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        out = self.conv1(x)

        # Apply instance norm conditionally
        if out.size(2) > 1 and out.size(3) > 1:
            out = self.norm1(out)
        out = self.relu(out)

        # Apply second reflection padding conditionally
        if out.size(2) > 1 and out.size(3) > 1:
            out = F.pad(out, (1, 1, 1, 1), mode='reflect')
        out = self.conv2(out)

        # Apply second instance norm conditionally
        if out.size(2) > 1 and out.size(3) > 1:
            out = self.norm2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out
        return out + x

    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)    

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)     

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.Conv2d(in_features, in_features, 3),

            nn.InstanceNorm2d(in_features)
        )

        # Assume ChannelAttention and CBAMBlock are previously defined
        self.se_block = ChannelAttention(in_features) 
        self.cse_block = CBAMBlock(in_features)

    def forward(self, x):
        identity = x  # Preserve the original input for the skip connection
        out = self.conv_block(x)
        out = self.se_block(out)  # Apply SE block to scale the conv_block output
        out = self.cse_block(out)  # Apply CBAM block to refine the feature maps
        
        
        out = F.relu(out)  # Apply ReLU or any other activation function if needed        


        out = out + identity  # Add the skip connection
        out = F.relu(out)  # Apply ReLU or any other activation function if needed
        
        
        return out


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        depth=1
        
        for _ in range(depth):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(depth):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)
        self.smoothing = TrainableSmoothingLayer(3)
        self.sharpening = TrainableSharpeningLayer(3)
        self.contrast=TrainableBrightnessContrastLayer(3)
        


    def forward(self, x):
        x=self.smoothing(x)
        x=self.sharpening(x)
        ## Above two added after lol+mixed ite2
        x=self.model(x)
        x=self.smoothing(x)
        x=self.sharpening(x)

        return x


class TrainableBrightnessContrastLayer(nn.Module):
    def __init__(self, in_channels):
        super(TrainableBrightnessContrastLayer, self).__init__()
        
        # Initialize alpha (contrast) for each channel
        self.alpha = nn.Parameter(torch.ones(in_channels))
        
        # Initialize beta (brightness) for each channel
        self.beta = nn.Parameter(torch.zeros(in_channels))
        
    def forward(self, x):
        # The alpha parameter scales the pixel values for contrast adjustment
        # The beta parameter shifts the pixel values for brightness adjustment
        # x is assumed to have shape (batch_size, in_channels, height, width)
        
        # The view method ensures the parameters have the correct shape for broadcasting
        alpha = self.alpha.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        
        # Apply the affine transformation for brightness and contrast
        return alpha * x + beta


class TrainableSmoothingLayer(nn.Module):
    def __init__(self, in_channels):
        super(TrainableSmoothingLayer, self).__init__()
        
        # Initialize with average kernel for smoothing
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
        
        # Initialize with a sharpening kernel
        initial_kernel = torch.tensor([[ 0.0,-1.0, 0.0],
                                       [-1.0, 5.0,-1.0],
                                       [ 0.0,-1.0, 0.0]])
                                       
        self.kernel = nn.Parameter(initial_kernel.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1))
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        
    def forward(self, x):
        self.conv.weight.data = self.kernel
        return self.conv(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)