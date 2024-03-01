import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np



## Need to test these two blocks
class CBAM(nn.Module):
    def __init__(self, in_features):
        super(CBAM, self).__init__()
        self.in_features = in_features
        self.conv1 = nn.Conv2d(in_features, in_features, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_features, in_features, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(in_features)
        self.ca = SFOM(in_features)
        self.sa = SPEM()

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

  
class SPEM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SPEM, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)    

class SFOM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SFOM, self).__init__()
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
            nn.Conv2d(in_features, in_features, 3),

            nn.InstanceNorm2d(in_features)
        )

        # Assume SFOM and CBAM are previously defined
        self.SFOM_layer = SFOM(in_features) 
        self.CBAM_layer = CBAM(in_features)

    def forward(self, x):
        identity = x  # Preserve the original input for the skip connection
        out = self.conv_block(x)
        out = self.SFOM_layer(out)  # Apply SE block to scale the conv_block output
        out = self.CBAM_layer(out)  # Apply CBAM block to refine the feature maps
        
        
        out = F.relu(out)  # Apply ReLU or any other activation function if needed        
        out = out + identity  # Add the skip connection
        out = F.relu(out)  # Apply ReLU or any other activation function if needed
        
        
        return out


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_GPSM=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        
        depth=1
        factor_=depth*2
        out_features = in_features*factor_
 
        for _ in range(depth):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*factor_

        # Residual blocks
        for _ in range(n_GPSM):
            model += [GPSM(in_features)]

        # Upsampling
        out_features = in_features//factor_
        for _ in range(depth):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//factor_

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)
        self.smoothing = TrainableSmoothingLayer(3)
        self.sharpening = TrainableSharpeningLayer(3)
        


    def forward(self, x):
        x=self.smoothing(x)
        x=self.sharpening(x)
        x=self.model(x)
        x=self.smoothing(x)        
        x=self.sharpening(x)

        return x


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