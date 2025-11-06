import torch
import torch.nn as nn
# from hybernet.encoders import get_encoder
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch.nn.functional as F
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):

        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class SAM_Module(nn.Module):
    def __init__(self, in_channels):
        super(SAM_Module, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_weights = torch.mean(x, dim=1, keepdim=True)
        attention_weights = self.sigmoid(attention_weights)
        x = x * attention_weights
        return x


class SE_Module(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SE_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        attention_weights = self.avg_pool(x).view(b, c)
        attention_weights = self.fc1(attention_weights)
        attention_weights = self.fc2(attention_weights)
        attention_weights = self.sigmoid(attention_weights).view(b, c, 1, 1)
        x = x * attention_weights
        return x

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, groups=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, groups=groups,bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

class DialtedSEFFN(nn.Module):

    def __init__(self, inplanes, outplanes, dilat, downsample=None, stride=1, t=1, scales=4, se=True, norm_layer=None):
        super(DialtedSEFFN, self).__init__()
        if inplanes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = inplanes * t
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales,
                                            groups=(bottleneck_planes // scales),dilation=dilat[i],
                                            padding=1*dilat[i]) for i in range(scales)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales)])
        self.conv3 = conv1x1(bottleneck_planes, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(outplanes) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out
class UPx2(nn.Module):

    def __init__(self, nIn, nOut):

        super().__init__()
        self.deconv = nn.ConvTranspose2d(nIn, nOut, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):

        output = self.deconv(input)
        output = self.bn(output)
        output = self.act(output)
        return output
    def fuseforward(self, input):
        output = self.deconv(input)
        output = self.act(output)
        return output


class DCR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv1 = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(nOut)
        self.act1 = nn.PReLU(nOut)

        # Additional convolutional layer
        self.conv2 = nn.Conv2d(nOut, nOut, (kSize, kSize), stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(nOut)
        self.act2 = nn.PReLU(nOut)

        # Add a 1x1 conv when the number of input and output channels are not equal
        self.downsample = None
        if nIn != nOut:
            self.downsample = nn.Sequential(
                nn.Conv2d(nIn, nOut, 1, stride=stride, bias=False),
                nn.BatchNorm2d(nOut),
            )

    def forward(self, input):
        residual = input

        out = self.conv1(input)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        out += residual
        out = self.act2(out)

        return out

class CB(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):

        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):

        output = self.conv(input)
        output = self.bn(output)
        return output

class C(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):

        super().__init__()
        padding = int((kSize - 1)/2)
        # print(nIn, nOut, (kSize, kSize))
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):

        output = self.conv(input)
        return output

class CDilated(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):

        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output

class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4],1)
        #combine_in_out = input + combine
        output = self.bn(combine)
        output = self.act(output)
        return output
    
class BR(nn.Module):

    def __init__(self, nOut):

        super().__init__()
        self.nOut=nOut
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):

        output = self.bn(input)
        # print("after bn :",output.size())
        output = self.act(output)
        # print("after act :",output.size())
        return output
    
class DilatedParllelResidualBlockB(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = max(int(nOut/5),1)
        n1 = max(nOut - 4*n,1)
        # print(nIn,n,n1,"--")
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1) # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2) # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4) # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8) # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16) # dilation rate of 2^4
        # print("nOut bf :",nOut)
        self.bn = BR(nOut)
        # print("nOut at :",self.bn.size())
        self.add = add

    def forward(self, input):

        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        #merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output

class InputProjectionA(nn.Module):

    def __init__(self, samplingTimes):

        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):

        for pool in self.pool:
            input = pool(input)
        return input
class RDFEncoder(nn.Module):

    def __init__(self, p=5, q=3):

        super().__init__()
        self.level1 = DCR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = DCR(16 + 3,19,3)
        self.level2_0 = DownSamplerB(16 +3, 64)
        self.fpe_block = DialtedSEFFN(inplanes=64, outplanes=64, dilat=[1, 2, 3, 4], se=True)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64 , 64))
        self.b2 = DCR(128 + 3,131,3)

        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128 , 128))
        self.b3 = DCR(256,32,3)
        
        self.sa = PAM_Module(32)
        self.sc = CAM_Module(32)
        
        self.sam = SAM_Module(32)
        self.se = SE_Module(32)
        
        self.conv_sa = DCR(32,32,3)
        self.conv_sc = DCR(32,32,3)
        self.conv_sam = DCR(32,32,3)
        self.conv_se = DCR(32,32,3)
        self.classifier = DCR(128, 32, 1, 1)


    def forward(self, input):

        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)
        output1_0 = self.fpe_block(output1_0)

        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1,  output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat) # down-sampled
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        cat_=torch.cat([output2_0, output2], 1)

        output2_cat = self.b3(cat_)
        out_sa=self.sa(output2_cat)
        out_sa=self.conv_sa(out_sa)

        out_sc=self.sc(output2_cat)
        out_sc=self.conv_sc(out_sc)

        out_sam=self.sam(output2_cat)
        out_sam=self.conv_sam(out_sam)

        out_se=self.se(output2_cat)
        out_se=self.conv_se(out_se)


        out_s=torch.cat([out_sa, out_sc, out_sam, out_se], dim=1)
        classifier = self.classifier(out_s)
        # print('classifier:', classifier.shape)
        return classifier
   

class RDFNet(nn.Module):


    def __init__(self, p=2, q=3, ):
        super().__init__()
        self.encoder = RDFEncoder(p, q)
        self.up_1_1 = UPx2(32,16)
        self.up_2_1 = UPx2(16,8)
        self.up_1_2 = UPx2(32,16)
        self.up_2_2 = UPx2(16,8)
        self.classifier_1 = UPx2(8,2)
        self.classifier_2 = UPx2(8,2)

    def forward(self, input):

        x=self.encoder(input)
        x1=self.up_1_1(x)
        x11=self.up_2_1(x1)
        classifier1=self.classifier_1(x11)
        x2=self.up_1_2(x)
        x22=self.up_2_2(x2)
        classifier2=self.classifier_2(x22)
        return (classifier1,classifier2)


if __name__ == "__main__":

    model = RDFNet()
    input_ = torch.randn((1, 3, 640, 360))
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    dring_area_seg, lane_line_seg = model(input_)
    
    print(dring_area_seg.shape)
    print(lane_line_seg.shape)
