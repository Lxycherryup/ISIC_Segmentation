
import torch
import torch.nn as nn
from backbone import resnet50
import numpy as np
import cv2
#计算热力图的方法
def save_feats_mean(x):
    b, c, h, w = x.shape
    if h == 256:
        with torch.no_grad():
            x = x.detach().cpu().numpy()
            x = np.transpose(x[0], (1, 2, 0))
            x = np.mean(x, axis=-1)
            x = x/np.max(x)
            x = x * 255.0
            x = x.astype(np.uint8)
            x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
            x = np.array(x, dtype=np.uint8)
            return x

#编码器、解码器中的残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        return x

#编码器模块 两倍下采样
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.r1 = ResidualBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.r1(inputs)
        p = self.pool(x)
        return x, p


#用于底部链接的transformer模块 作用是增强模型的全局上下文建模能力  增加模型的鲁棒性
class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, dim, num_layers=2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.tblock = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        b, c, h, w = x.shape
        x = x.reshape((b, c, h*w))
        x = self.tblock(x)
        x = x.reshape((b, c, h, w))
        x = self.conv2(x)
        return x


#底部空洞卷积模块  作用是扩大感受野
class DilatedConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=9, dilation=9),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c5 = nn.Sequential(
            nn.Conv2d(out_c*4, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.c5(x)
        return x

#解码器模块  两个输入  一个时底部的输入 一个是跳跃链接的输入  输出结果是对输入的两倍上采样
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        #编码器使用双向性插值进行两倍上采样
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        #使用两个残差块进行特征提取
        self.r1 = ResidualBlock(in_c[0]+in_c[1], out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r1(x)
        x = self.r2(x)
        return x

#GAM Attention
#注意力机制，作用是对特征进行加权，消除冗余的特征
class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out



class TResUnet(nn.Module):
    def __init__(self):
        super().__init__()

        """ ResNet50 """
        #编码器使用resnet50作为骨干网络
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        """ Bridge blocks """
        #transformer
        self.b1 = Bottleneck(1024, 256, 256, num_layers=2)
        #空洞卷积
        self.b2 = DilatedConv(1024, 256)

        """ Decoder """
        self.d1 = DecoderBlock([512, 512], 256)
        self.d2 = DecoderBlock([256, 256], 128)
        self.d3 = DecoderBlock([128, 64], 64)
        self.d4 = DecoderBlock([64, 3], 32)

        """GAM Attention"""
        self.gam1 = GAM_Attention(in_channels=64,out_channels=64,rate=4)
        self.gam2 = GAM_Attention(in_channels=256,out_channels=256,rate=4)
        self.gam3 = GAM_Attention(in_channels=512,out_channels=512,rate=2)

        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, heatmap=None):
        s0 = x
        s1 = self.layer0(s0)    ## [-1, 64, h/2, w/2]
        s1_1 = s1
        s2 = self.layer1(s1)    ## [-1, 256, h/4, w/4]
        s2_1 = s2
        s3 = self.layer2(s2)    ## [-1, 512, h/8, w/8]
        s3_1 = s3
        s4 = self.layer3(s3)    ## [-1, 1024, h/16, w/16]


        b1 = self.b1(s4)
        b2 = self.b2(s4)
        b3 = torch.cat([b1, b2], axis=1)

        # sikp connection use gam attention
        s1_1 = self.gam1(s1_1)
        s2_1 = self.gam2(s2_1)
        s3_1 = self.gam3(s3_1)

        d1 = self.d1(b3, s3_1)
        d2 = self.d2(d1, s2_1)
        d3 = self.d3(d2, s1_1)
        d4 = self.d4(d3, s0)

        y = self.output(d4)

        if heatmap != None:
            hmap = save_feats_mean(d4)
            return hmap, y
        else:
            return y

if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    model = TResUnet()
    y = model(x)
    print(y.shape)