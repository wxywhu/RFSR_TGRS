#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
      
class Res3DBlock(nn.Module):
    def __init__(self, n_feats, bias=True, act=nn.ReLU(True), res_scale=1):
        super(Res3DBlock, self).__init__()
        
        self.body = nn.Sequential(nn.Conv3d(1, n_feats, (3,1,1),1,(1,0,0), bias=bias),
                                  act,
                                  nn.Conv3d(n_feats, 1, (1,3,3),1,(0,1,1), bias=bias)                   
                    )
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x.unsqueeze(1))+x.unsqueeze(1)
        return x.squeeze(1)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(
        self, n_feat,reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, 3,1,1, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3,1,1,bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 4 * n_feats, 3,1,1,bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)




class ShuffleDown(nn.Module):
    def __init__(self, scale):
        super(ShuffleDown, self).__init__()
        self.scale = scale

    def forward(self, x):
        b, cin, hin, win= x.size()
        cout = cin * self.scale ** 2
        hout = hin // self.scale
        wout = win // self.scale
        output = x.view(b, cin, hout, self.scale, wout, self.scale)
        output = output.permute(0, 1, 5, 3, 2, 4).contiguous()
        output = output.view(b, cout, hout, wout)
        return output
        
class Net(nn.Module):
    def __init__(self, scale, seq_len, devices):
        super(Net, self).__init__()
        self.n_feats = 64
        self.kernel_size = 3 
        self.devices = devices
        self.sub=16
        self.scale = scale

        self.g = 8

        
        self.layer1 =  default_conv(self.sub+self.n_feats+self.sub*self.scale ** 2, self.n_feats, self.kernel_size)

        self.out_layer1 = default_conv(self.n_feats, self.sub,self.kernel_size)
        self.out_layer2 = default_conv(self.n_feats, self.n_feats, self.kernel_size)


        n_a=6
        body1  = [RCAB(self.n_feats) for _ in range(n_a)]
        self.RB1 = nn.Sequential(*body1)
        self.up = Upsampler(self.scale, self.n_feats)
        self.down = ShuffleDown(self.scale)
        
        self.act = nn.ReLU(True)
        n_b=3
        body2  = [Res3DBlock(seq_len) for _ in range(n_b)]
        self.body2 = nn.Sequential(*body2)
        
    def forward(self, x):
        out = []
        B,C,h,w =x.shape

        #p=self.sub-C%self.sub
        #ini = torch.zeros(B,p,h,w).to(self.devices)
        #x=torch.cat([x,ini],1)
        
        h1 = torch.zeros(B,self.n_feats,h,w).to(self.devices)
        sr = torch.zeros(B,self.sub*self.scale ** 2,h,w).to(self.devices)

        for x_ilr in torch.chunk(x, self.g, 1):
          h1 = self.act(self.layer1(torch.cat([h1,sr,x_ilr], dim=1)))
          h1 = self.RB1(h1)
          sr = self.out_layer1(self.up(h1)) + F.interpolate(x_ilr,(h*self.scale,w*self.scale))
          h1 = self.out_layer2(h1)
          out.append(sr)
          sr = self.down(sr)
          
        out = torch.cat(out[:],1)[:,0:C,:,:]
        out = self.body2(out)
        return out
        
        




