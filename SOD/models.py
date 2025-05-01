import torch
import torch.nn.functional as F
import torch.nn as nn
import math
interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)
import torch
import torch.nn as nn
import torch.nn.functional as F

class convbnrelu(nn.Module):  
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
    
class CNL(nn.Module):
    def __init__(self, in_channel=3,out_channel=6,in_num=1,out_num=3,dl=[1,3,7],pool=False,up=False,ke=3,drop=False,drop_r=0.1,downrate=2,uprate=2):
        super(CNL, self).__init__()
        self.out=out_num
        self.inn=in_num
        self.out_c=out_channel
        self.poo=pool
        self.upp=up
        self.model_list = nn.ModuleList([nn.ModuleList([None for _ in range(out_num)]) for _ in range(in_num)])

        for i in range(in_num):
            for j in range(out_num):
                self.model_list[i][j] = nn.Sequential(
                    convbnrelu(in_channel, in_channel, k=ke, s=1, p=int((ke-1)/2)*dl[j], d=dl[j], g=in_channel),
                    convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=True),


                )
                if pool==True:
                    self.model_list[i][j].append(nn.MaxPool2d(kernel_size=downrate, stride=downrate))
                if up==True:
                    self.model_list[i][j].append(
                        nn.Upsample(scale_factor=uprate, mode='bilinear', align_corners=True)

                    )
                if drop == True:
                    self.model_list[i][j].append(
                        nn.Dropout2d(drop_r)
                    )


        self.cat=nn.ModuleList([nn.Conv2d(out_channel*in_num, out_channel, kernel_size=1) for _ in range(out_num)])

    def forward(self, x):

        for i in range(self.out):
            for o in range(self.inn):
                now = self.model_list[o][i](x[..., o])
                if o==0:
                    s=now
                else:
                    s=torch.cat((s,now), dim=1)
            s=self.cat[i](s).unsqueeze(4)

            if i==0:
                s1=s
            else:
                s1=s=torch.cat((s1,s), dim=4)
        return s1
    


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.l0 = CNL(in_channel=3, out_channel=16, in_num=1, out_num=3, dl=[1, 4, 1], pool=False,
                      up=False,
                      ke=5)
        self.l1 = CNL(in_channel=16, out_channel=32, in_num=3, out_num=5, dl=[1,4,12,36,108], pool=True,
                      up=False)
        self.z0 = CNL(in_channel=32, out_channel=32, in_num=5, out_num=1, dl=[1,4,12,36,108], pool=True,
                      up=False)
        self.l2 = CNL(in_channel=32, out_channel=64, in_num=5, out_num=3, dl=[1,4,12,36,108], pool=True,
                      up=False)
        self.l3 = CNL(in_channel=64, out_channel=128, in_num=3, out_num=2, dl=[1,4,12,36,108], pool=True,
                      up=False)
        self.l4 = CNL(in_channel=128, out_channel=256, in_num=2, out_num=1, dl=[1,4,12,36,108], pool=True,
                      up=False)
        self.lz0 = CNL(in_channel=64, out_channel=64, in_num=3, out_num=1, dl=[1,4,12,36,108], pool=True,
                       up=False)

    def forward(self, x):
        x1 = x.unsqueeze(4)
        x1 = self.l0(x1)
        x1 = self.l1(x1)
        b1 = self.z0(x1)
        x1 = self.l2(x1)
        b2 = self.lz0(x1)
        x1 = self.l3(x1)
        x1 = self.l4(x1)#
        bm = x1

        return b1, b2, bm

class LMFNet(nn.Module):
    def __init__(self):
        super(LMFNet, self).__init__()
        self.encoder = encoder()
        self.z1 = CNL(in_channel=32, out_channel=32, in_num=1, out_num=3, dl=[1,4,12,36,108], pool=False,up=True,drop=False)
        self.lz1 = CNL(in_channel=64, out_channel=64, in_num=1, out_num=3, dl=[1,4,12,36,108], pool=False,up=True,drop=False)
        self.l5 = CNL(in_channel=256, out_channel=128, in_num=1, out_num=2, dl=[1,4,12,36,108], pool=False,up=True,drop=False)
        self.l6 = CNL(in_channel=128, out_channel=64, in_num=2, out_num=3, dl=[1,4,12,36,108], pool=False,up=True,drop=True)
        self.l7 = CNL(in_channel=128, out_channel=32, in_num=3, out_num=3, dl=[1,4,12,36,108], pool=False,up=True,drop=True)
        self.l8 = CNL(in_channel=64, out_channel=16, in_num=3, out_num=1, dl=[1,4,12,36,108], pool=False,up=True,drop=True)
        self.F = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b1, b2, bm = self.encoder(x)
        xz0 = b1
        xz0 = self.z1(xz0)
        xz = b2
        xz = self.lz1(xz)
        x1 = bm
        x1 = self.l5(x1)
        x1 = self.l6(x1)
        x1 = self.l7(torch.cat((x1, xz), dim=1))
        x1 = self.l8(torch.cat((x1, xz0), dim=1))
        x1 = x1.squeeze(4)
        x1 = self.F(x1)
        x1 = self.sigmoid(x1)
        return x1
   
    
    
 
    
    
    
    
    
    
    

   


