import torch
import torch.nn as nn
import torch.nn.functional as F

class APSTF(nn.Module):
    """Adaptive Parameterized Soft Thresholding Function (with improved ECA module)"""
    def __init__(self, channel, reduction=16, dim=2):
        super().__init__()
        self.dim = dim  # 2D or 3D input
        self.channel = channel
        
        # --- Improved ECA module ---
        # Global average pooling (supports 2D/3D)
        if self.dim == 2:
            self.gap = nn.AdaptiveAvgPool2d(1)
        elif self.dim == 3:
            self.gap = nn.AdaptiveAvgPool3d(1)
        
        # Channel attention (1D convolution for scaling factor α)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()  # Constrain α to [0,1] range
        )

    def forward(self, x):
        # --- Dynamically generate threshold τ ---
        if self.dim == 2:
            # 2D input (B, C, H, W)
            b, c, h, w = x.size()
            gap = self.gap(x).view(b, c)  # [B, C]
        else:
            # 3D input (B, C, D, H, W)
            b, c, d, h, w = x.size()
            gap = self.gap(x).view(b, c)  # [B, C]
        
        # Generate scaling factor α via 1D convolution
        alpha = self.conv(gap.unsqueeze(1))  # [B, 1, C]
        alpha = alpha.squeeze(1)  # [B, C]
        
        # Calculate threshold τ = α * mean of feature map absolute values
        if self.dim == 2:
            avg_abs = x.abs().mean(dim=(2, 3))  # [B, C]
        else:
            avg_abs = x.abs().mean(dim=(2, 3, 4))  # [B, C]
        tau = alpha * avg_abs  # [B, C]
        tau = tau.view(b, c, *([1] * (self.dim)))  # Expand dimensions, e.g., 2D: [B, C, 1, 1]
        
        # --- Apply soft thresholding function ---
        # Positive processing: y_pos = (x - τ) if x > τ, else 0
        y_pos = torch.where(x > tau, x - tau, torch.zeros_like(x))
        # Negative processing: y_neg = (x + τ) if x < -τ, else 0
        y_neg = torch.where(x < -tau, x + tau, torch.zeros_like(x))
        # Combine features
        y = y_pos + y_neg
        return y

class ConvBlock(nn.Module):
    """Convolutional block (Conv+BN+APSTF)×2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            APSTF(channel=out_channels, dim=2),  # Replace ReLU with APSTF
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            APSTF(channel=out_channels, dim=2)   # Replace ReLU with APSTF
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    """Upsampling block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            APSTF(channel=out_channels, dim=2)  # Replace ReLU with APSTF
        )

    def forward(self, x):
        return self.up(x)

class UNetPlusPlus(nn.Module):
    """U-Net++ architecture (APSTF activation version)"""
    def __init__(self, in_channels=6, out_channels=24, features=[32, 64, 128, 256]):
        super().__init__()
        self.out_channels = out_channels
        
        # Encoder part
        self.pool = nn.MaxPool2d(2)
        
        # Layer 0
        self.conv0_0 = ConvBlock(in_channels, features[0])
        
        # Layer 1
        self.conv1_0 = ConvBlock(features[0], features[1])
        
        # Layer 2
        self.conv2_0 = ConvBlock(features[1], features[2])
        
        # Layer 3
        self.conv3_0 = ConvBlock(features[2], features[3])
        
        # Layer 4 (bottleneck)
        self.conv4_0 = ConvBlock(features[3], features[3] * 2)
        
        # Upsampling layers
        self.up1 = UpConv(features[3] * 2, features[3])  # 512 -> 256
        self.up2 = UpConv(features[3], features[2])      # 256 -> 128
        self.up3 = UpConv(features[2], features[1])      # 128 -> 64
        self.up4 = UpConv(features[1], features[0])      # 64 -> 32
        
        # Fixed dense connections - ensure correct channel calculations
        
        # Layer 3 other nodes
        # Input: x3_0 (256) + up1(x4_0) (256) = 512 channels
        self.conv3_1 = ConvBlock(features[3] + features[3], features[3])  # 512 -> 256
        
        # Layer 2 other nodes
        # conv2_1: x2_0 (128) + up2(x3_0) (128) = 256 channels
        self.conv2_1 = ConvBlock(features[2] + features[2], features[2])  # 256 -> 128
        
        # conv2_2: x2_0 (128) + x2_1 (128) + up2(x3_1) (128) = 384 channels
        self.conv2_2 = ConvBlock(features[2] * 3, features[2])  # 384 -> 128
        
        # Layer 1 other nodes
        # conv1_1: x1_0 (64) + up3(x2_0) (64) = 128 channels
        self.conv1_1 = ConvBlock(features[1] + features[1], features[1])  # 128 -> 64
        
        # conv1_2: x1_0 (64) + x1_1 (64) + up3(x2_1) (64) = 192 channels
        self.conv1_2 = ConvBlock(features[1] * 3, features[1])  # 192 -> 64
        
        # conv1_3: x1_0 (64) + x1_1 (64) + x1_2 (64) + up3(x2_2) (64) = 256 channels
        self.conv1_3 = ConvBlock(features[1] * 4, features[1])  # 256 -> 64
        
        # Layer 0 other nodes
        # conv0_1: x0_0 (32) + up4(x1_0) (32) = 64 channels
        self.conv0_1 = ConvBlock(features[0] + features[0], features[0])  # 64 -> 32
        
        # conv0_2: x0_0 (32) + x0_1 (32) + up4(x1_1) (32) = 96 channels
        self.conv0_2 = ConvBlock(features[0] * 3, features[0])  # 96 -> 32
        
        # conv0_3: x0_0 (32) + x0_1 (32) + x0_2 (32) + up4(x1_2) (32) = 128 channels
        self.conv0_3 = ConvBlock(features[0] * 4, features[0])  # 128 -> 32
        
        # conv0_4: x0_0 (32) + x0_1 (32) + x0_2 (32) + x0_3 (32) + up4(x1_3) (32) = 160 channels
        self.conv0_4 = ConvBlock(features[0] * 5, features[0])  # 160 -> 32
        
        # Output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.final_act = nn.Sigmoid()  
        self.size_adjust = nn.AdaptiveAvgPool2d((24, 24))

    def forward(self, x):
        # Encoder path
        x0_0 = self.conv0_0(x)                    # [B, 32, 25, 25]
        
        x1_0 = self.conv1_0(self.pool(x0_0))      # [B, 64, 12, 12]
        
        x2_0 = self.conv2_0(self.pool(x1_0))      # [B, 128, 6, 6]
        
        x3_0 = self.conv3_0(self.pool(x2_0))      # [B, 256, 3, 3]
        
        x4_0 = self.conv4_0(self.pool(x3_0))      # [B, 512, 1, 1]
        
        # Decoder path - U-Net++ dense connections
        
        # Layer 3
        up1 = self.up1(x4_0)                      # [B, 256, 2, 2]
        up1_resized = F.interpolate(up1, size=(3, 3), mode='bilinear', align_corners=False)
        x3_1 = self.conv3_1(torch.cat([x3_0, up1_resized], dim=1))  # [B, 256, 3, 3]
        
        # Layer 2
        up2_0 = self.up2(x3_0)                    # [B, 128, 6, 6]
        x2_1 = self.conv2_1(torch.cat([x2_0, up2_0], dim=1))        # [B, 128, 6, 6]
        
        up2_1 = self.up2(x3_1)                    # [B, 128, 6, 6]
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, up2_1], dim=1))  # [B, 128, 6, 6]
        
        # Layer 1
        up3_0 = self.up3(x2_0)                    # [B, 64, 12, 12]
        x1_1 = self.conv1_1(torch.cat([x1_0, up3_0], dim=1))        # [B, 64, 12, 12]
        
        up3_1 = self.up3(x2_1)                    # [B, 64, 12, 12]
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, up3_1], dim=1))  # [B, 64, 12, 12]
        
        up3_2 = self.up3(x2_2)                    # [B, 64, 12, 12]
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, up3_2], dim=1))  # [B, 64, 12, 12]
        
        # Layer 0
        up4_0 = self.up4(x1_0)                    # [B, 32, 24, 24]
        up4_0_resized = F.interpolate(up4_0, size=(25, 25), mode='bilinear', align_corners=False)
        x0_1 = self.conv0_1(torch.cat([x0_0, up4_0_resized], dim=1))  # [B, 32, 25, 25]
        
        up4_1 = self.up4(x1_1)                    # [B, 32, 24, 24]
        up4_1_resized = F.interpolate(up4_1, size=(25, 25), mode='bilinear', align_corners=False)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, up4_1_resized], dim=1))  # [B, 32, 25, 25]
        
        up4_2 = self.up4(x1_2)                    # [B, 32, 24, 24]
        up4_2_resized = F.interpolate(up4_2, size=(25, 25), mode='bilinear', align_corners=False)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, up4_2_resized], dim=1))  # [B, 32, 25, 25]
        
        up4_3 = self.up4(x1_3)                    # [B, 32, 24, 24]
        up4_3_resized = F.interpolate(up4_3, size=(25, 25), mode='bilinear', align_corners=False)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, up4_3_resized], dim=1))  # [B, 32, 25, 25]
        
        # Output layer
        out = self.final_conv(x0_4)               # [B, 24, 25, 25]
        out = self.size_adjust(out)               # [B, 24, 24, 24]
        out = self.final_act(out)                 
        out = out.unsqueeze(1)                    # [B, 1, 24, 24, 24]
        
        return out