import torch
import torch.nn as nn
import torchvision.models as models

class ResNetUNet(nn.Module):
    def __init__(self, n_classes=1, pretrained=True):
        super().__init__()

        # --- Encoder: ResNet34 ---
        base_model = models.resnet34(pretrained=pretrained)
        self.base_layers = list(base_model.children())

        # take stages of ResNet
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # conv1 + bn + relu
        self.layer0_pool = self.base_layers[3]              # maxpool
        self.layer1 = self.base_layers[4]                   # conv2_x
        self.layer2 = self.base_layers[5]                   # conv3_x
        self.layer3 = self.base_layers[6]                   # conv4_x
        self.layer4 = self.base_layers[7]                   # conv5_x

        # --- Decoder ---
        self.up4 = self._up_block(512, 256)
        self.up3 = self._up_block(256+256, 128)
        self.up2 = self._up_block(128+128, 64)
        self.up1 = self._up_block(64+64, 64)
        self.up0 = self._up_block(64+64, 32)

        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder forward
        x0 = self.layer0(x)           # 64
        x0_pool = self.layer0_pool(x0)
        x1 = self.layer1(x0_pool)     # 64
        x2 = self.layer2(x1)          # 128
        x3 = self.layer3(x2)          # 256
        x4 = self.layer4(x3)          # 512

        # Decoder with skip connections
        d4 = self.up4(x4)
        d4 = torch.cat([d4, x3], dim=1)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, x2], dim=1)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x1], dim=1)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x0], dim=1)

        d0 = self.up0(d1)

        out = self.final_conv(d0)  # logits
        return out