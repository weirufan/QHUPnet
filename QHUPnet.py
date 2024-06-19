from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
#from test_bottleneck import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class Conv_block(nn.Module):
    """
    Convolution Block #卷积块
    """
    def __init__(self, in_ch, out_ch):
        super(Conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            #nn.GroupNorm(16,out_ch),
            nn.Hardswish(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            #nn.GroupNorm(16,out_ch),
            nn.Hardswish(inplace=True))
        self.conv_1x1_output = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x):

        y = self.conv(x)
        x = self.conv_1x1_output(x)
        return x+y


class Dense_layer(nn.Sequential):
    def __init__(self,in_ch,bottleneck_size,growth_rate):
    
        super(Dense_layer,self).__init__()
        
        self.add_module('norm1', nn.BatchNorm2d(in_ch)),
        self.add_module('relu1', nn.Hardswish(inplace=True)),
        self.add_module('conv1', nn.Conv2d(in_ch, bottleneck_size, kernel_size=1, stride=1, padding=0, bias=True)),
        self.add_module('norm2', nn.BatchNorm2d(bottleneck_size)),
        self.add_module('relu2', nn.Hardswish(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bottleneck_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=True)),

    def forward(self, x):
        new_features = super(Dense_layer, self).forward(x)
        return torch.cat([x, new_features], 1)

class Dense_block(nn.Sequential):
            
    def __init__(self,in_channels,layer_counts,growth_rate):
        super(Dense_block,self).__init__()
        for i in range(layer_counts):
            bottleneck_size = 4*growth_rate
            layer = Dense_layer(in_channels + i * growth_rate, bottleneck_size, growth_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class TransitionLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.Hardswish(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,kernel_size=1, stride=1, bias=False))

class Conv_pooling(nn.Module):
    """
    # conv->pooling        #output_shape = (image_shape-filter_shape+2*padding)/stride + 1
    """
    def __init__(self, in_ch, out_ch):
        super(Conv_pooling, self).__init__()

        self.ConvPooling = nn.Sequential(
            #nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=1, padding=1,dilation=2, bias=True),
            #nn.BatchNorm2d(in_ch),
            #nn.Hardswish(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.Hardswish(inplace=True)
        )

    def forward(self, x):
        out=self.ConvPooling(x)
        return out


class ASPP(nn.Module):
    
    def __init__(self, in_channel):
        super(ASPP,self).__init__()
          
        self.atrous_block1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1)
        self.atrous_block2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=2, dilation=2)
        self.atrous_block3 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=3, dilation=3)
        self.atrous_block4 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=4, dilation=4)
        self.conv_1x1_output = nn.Conv2d(in_channel * 4, in_channel, kernel_size=1, stride=1)
        self.ConvPooling = nn.Sequential(
            nn.Conv2d(in_channel, in_channel*2, kernel_size=2, stride=1, padding=1, dilation=2, bias=True),
            nn.BatchNorm2d(in_channel*2),
            nn.Hardswish(inplace=True),
            #nn.Conv2d(in_channel, in_channel, kernel_size=2, stride=2, padding=0, bias=True),
            nn.MaxPool2d(2,stride=2),
            nn.BatchNorm2d(in_channel*2),
            nn.Hardswish(inplace=True)
        )

    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block2 = self.atrous_block2(x)
        atrous_block3 = self.atrous_block3(x)
        atrous_block4 = self.atrous_block4(x)
        out = self.conv_1x1_output(torch.cat([ atrous_block1, atrous_block2, atrous_block3, atrous_block4], dim=1))
        out = self.ConvPooling(out+x)
        return out

class UpConv_block(nn.Module):
    """
    Convolution Block #卷积块
    """
    def __init__(self, in_ch, out_ch):
        super(UpConv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.Hardswish(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.Hardswish(inplace=True))

    def forward(self, x):

        y = self.conv(x)
        
        return y

class CatConv_block(nn.Module):
    """
    Convolution Block #卷积块
    """
    def __init__(self, in_ch, growth):
        super(CatConv_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch*growth, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.Hardswish(inplace=True))
        
        self.Up=nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.BatchNorm2d(in_ch*2),
            nn.Hardswish(inplace=True),
            nn.Conv2d(in_ch*2, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.Hardswish(inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch*growth, in_ch*growth, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_ch*growth),
            nn.Hardswish(inplace=True),
            nn.Conv2d(in_ch*growth, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.Hardswish(inplace=True)
            )

    def forward(self, x1, x2):
        
        x2 = self.Up(x2)
        x = torch.cat([x1,x2],dim=1)
        y = self.conv2(x)+self.conv1(x)
        
        return y

class Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=2, out_ch=1):
        super(Net, self).__init__()

        #self.swish=nn.SiLU(inplace=True)
        #self.swish=nn.Hardswish(inplace=True)
        #self.act=nn.Sigmoid()
        #self.Conv0=Conv_block(1,32)
        self.CP=Conv_pooling(1,32)
        self.Aspp1=ASPP(32)
        self.Aspp2=ASPP(64)
        self.Aspp3=ASPP(128)
        self.Aspp4=ASPP(256)
        self.Aspp5=ASPP(512)
        #self.DP=nn.Dropout(0.2)
        #self.Conv00=Conv_block(1,32)
        self.conv1x1=nn.Conv2d(16, 1, kernel_size=1, stride=1)     
        self.Conv=Conv_block(32,16)
        
        self.Dense1=Dense_block(512,5,32)
        self.Dense2=Dense_block(256,5,16)
        self.Dense3=Dense_block(128,5,16)
        self.Dense4=Dense_block(64,5,16)
        self.Dense5=Dense_block(32,5,16)
        
        self.Trans1=TransitionLayer(512+5*32,512)
        self.Trans2=TransitionLayer(256+5*16,256)
        self.Trans3=TransitionLayer(128+5*16,128)
        self.Trans4=TransitionLayer(64+5*16,64)
        self.Trans5=TransitionLayer(32+5*16,32)
        
        self.CatConv01=CatConv_block(32,2)
        self.CatConv11=CatConv_block(64,2)
        self.CatConv21=CatConv_block(128,2)
        self.CatConv31=CatConv_block(256,2)
        self.CatConv41=CatConv_block(512,2)
        
        self.CatConv02=CatConv_block(512,2)
        self.CatConv12=CatConv_block(256,2)
        self.CatConv22=CatConv_block(128,2)
        self.CatConv32=CatConv_block(64,2)
        self.CatConv42=CatConv_block(32,2)
        
        #self.DP1=nn.Dropout2d(0.4,inplace=True)
        #self.DP2=nn.Dropout2d(0.4,inplace=True)
        
    def forward(self, x):
        
        #Conv00=self.Conv0(x)  #1->32  512->512
        #Conv00=self.CP(x)  #32->32  256->256
        Conv00=self.CP(x)  #1->32  512->256
        Conv10=self.Aspp1(Conv00) #32->64 256->128
        Conv20=self.Aspp2(Conv10)   #64->128 128->64
        Conv30=self.Aspp3(Conv20)   #128->256  64->32
        Conv40=self.Aspp4(Conv30)   #256->512 32->16
        #Conv40=self.DP1(Conv40)
        Conv50=self.Aspp5(Conv40)   #512->1024 16->8
        #Conv50=self.DP2(Conv50)
        
        Conv01=self.CatConv01(Conv00,Conv10) #32->32  256->256
        Conv11=self.CatConv11(Conv10,Conv20) #64->64  128->128
        Conv21=self.CatConv21(Conv20,Conv30) #128->128  64->64
        Conv31=self.CatConv31(Conv30,Conv40) #256->256  32->32
        Conv41=self.CatConv41(Conv40,Conv50) #512->512  16->16
        
        Conv42=self.CatConv02(Conv41,Conv50) #512->512  16->16
        Conv42=self.Dense1(Conv42)
        Conv42=self.Trans1(Conv42)
        
        Conv32=self.CatConv12(Conv31,Conv42) #256->256  32->32
        Conv32=self.Dense2(Conv32)
        Conv32=self.Trans2(Conv32)
        
        Conv22=self.CatConv22(Conv21,Conv32) #128->128  64->64
        Conv22=self.Dense3(Conv22)
        Conv22=self.Trans3(Conv22)
        
        Conv12=self.CatConv32(Conv11,Conv22) #64->64  128->128
        Conv12=self.Dense4(Conv12)
        Conv12=self.Trans4(Conv12)
        
        Conv02=self.CatConv42(Conv01,Conv12) #32->32  256->256
        Conv02=self.Dense5(Conv02)
        Conv02=self.Trans5(Conv02)

        Conv0=self.Conv(Conv02) #32->16 256->256
        out=self.conv1x1(Conv0) #16->1 256->256

        return out
