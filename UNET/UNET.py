import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np




class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DilatedConv(nn.Module):
    def __init__(self,in_channels,out_channels,dilation = [10,5]):
        super(DilatedConv,self).__init__()
        self.dilated = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, dilation=dilation[0] if in_channels == 64 else dilation[1],bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, dilation=dilation[0] if in_channels == 64 else dilation[1],bias=False),
        )
    def forward(self,x):
        return self.dilated(x)


class DoubleConv1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv1,self).__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, dilation=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, dilation=1,bias=False),
        )
    def forward(self,x):
        return self.doubleconv(x)


class CNN(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CNN,self).__init__()
        self.cnn     =  nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.cnn(x)

class FullConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(FullConv,self).__init__()
        self.in_channels =  in_channels
        self.dilated     =  DilatedConv(in_channels,out_channels)
        self.doubleCONV  =  DoubleConv1(in_channels,out_channels)
        self.DilatedCnn  =  CNN(in_channels=out_channels*2,out_channels=(out_channels))
        self.cnn         =  CNN(in_channels,out_channels)
    def forward(self,x):
        # if self.in_channels <= 128:
        #     x2    = self.doubleCONV(x)
        #     x1    = self.dilated(x)
        #     zeros = torch.zeros((x2.shape)).to('cuda')
        #     zeros[:, :, :x1.shape[2], :x1.shape[3]] = x1
        #     x = torch.cat((zeros,x2),dim=1)
        #     del zeros
        #     x = self.DilatedCnn(x)
        # else:
        #     x = self.cnn(x)

        return self.cnn(x)


class UNET(nn.Module):
    def __init__(
            self,in_channels=3,out_channels=1,features=[64,128,256,512]
    ):
        super(UNET,self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.D_ups = nn.ModuleList()
        self.D_downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.softmax = nn.Softmax2d()

        #Down part of UNET
        for feature in features:

            self.downs.append(FullConv(in_channels,feature))
            in_channels = feature


        #Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2,feature,kernel_size=2,stride=2
                )
            )
            self.ups.append(FullConv(feature*2,feature))

        self.bottlneck = DoubleConv(features[-1],features[-1]*2)
        self.final_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)

    def forward(self,x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottlneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0,len(self.ups),2):

            x = self.ups[idx](x)

            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x,size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection,x),dim=1)
            x = self.ups[idx+1](concat_skip)





        return self.final_conv(x)



