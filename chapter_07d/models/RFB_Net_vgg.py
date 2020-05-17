import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4


        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)# 张量拼接
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

class RFBNet(nn.Module):
    """RFB Net for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1711.07767.pdf for more details on RFB Net.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    # RFBNet(phase, size, *multibox(size, vgg(base[str(size)], 3),
    #                             add_extras(size, extras[str(size)], 1024),
    #                             mbox[str(size)], num_classes), num_classes)

    # multibox (size, vgg, extra_layers, cfg, num_classes)
    # multibox return : vgg, extra_layers, (loc_layers, conf_layers)
    def __init__(self, phase, size, base, extras, head, num_classes,debug_= False):
        super(RFBNet, self).__init__()
        self.phase = phase # train/test
        self.num_classes = num_classes# voc num_class = 21
        self.size = size
        self.debug_ = debug_
        if self.debug_:
            print('\n/***************** RFBNet Build ******************/')
            print('head (loc & conf): ',len(head))

            for i in range(len(head)):
                print(i,'> len : ',len(head[i]))
                if i == 0:
                    print('loc ',mbox[str(size)],' * 4',)
                else:
                    print('conf ',mbox[str(size)],' * 21',)
                for j in range(len(head[i])):
                    print('    %s) '%(j),head[i][j])

        if size == 300:
            self.indicator = 3
        else:
            print("Error: Sorry only SSD300 are supported!")
            return
        # vgg network
        self.base = nn.ModuleList(base)
        # conv_4_3
        self.Norm = BasicRFB_a(512,512,stride = 1,scale=1.0)
        print('\n BasicRFB_a ： \n',self.Norm)
        self.extras = nn.ModuleList(extras) # extras 主路径序列

        # head : loc_layers, conf_layers
        self.loc = nn.ModuleList(head[0])# bbox
        self.conf = nn.ModuleList(head[1])# classify
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        s = self.Norm(x)#self.Norm = BasicRFB_a(512,512,stride = 1,scale=1.0)
        if self.debug_:
            print('  \n --------------------->> feature map')
            print('step1 : feature map 0 : ',s.size())
        sources.append(s)# 第一个 feature map

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)

        # 计算 feature maps
        idx_feature = 1
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k%2 ==0:
                if self.debug_:
                    print('step1 : feature map %s : '%(idx_feature),x.size())
                idx_feature += 1
                sources.append(x)

        # apply multibox head to source layers --->>
        if self.debug_:
            print('\n -------------->> feature map -->> chage to -->> loc & conf ')
        idx_feature = 0
        for (x, l, c) in zip(sources, self.loc, self.conf): # self.loc, self.conf : 2 head
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            if self.debug_:
                print('----------------------')
                print('step2 : feature map loc %s : '%(idx_feature),l(x).size())
                print('step2 : feature map conf %s : '%(idx_feature),c(x).size())
                idx_feature += 1

        #print([o.size() for o in loc])
        if self.debug_:
            print('  \n --------------------->> loc conf  ','all box num : ',11620)
            for i in range(len(loc)):
                print(' > <%s> loc : '%(i),loc[i].size(),' default box num : ',mbox['300'][i],' num_classes ',self.num_classes)
                print(' >    conf : ',conf[i].size())

        # concate所有检测分支上的检测结果
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.debug_:
            print('  \n --------------------->> cat || bbox : 11620 * 4 = 46480, confidence : 11620 * 21(num_class) = 244020 ')
            for i in range(len(loc)):
                print('* batch size %s '%(i))
                print(' >> <%s> loc : '%(i),loc[i].size(),' default box num : ',mbox['300'][i],' num_classes ',self.num_classes)
                print(' >>    conf : ',conf[i].size())

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                conf.view(conf.size(0), -1, self.num_classes),  # conf preds
            )
        return output

    def load_weights(self, base_file):
        """
        load basenet weights
        """
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

# vgg16 backbone，字典格式存储
base = {'300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',512, 512, 512]}
# vgg(base[str(size)], 3)
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i # 输入通道
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':#ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:# inplace=True ： 计算结果不会有影响。利用in-place计算可以节省内（显）存
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # zero-padding 卷积之前补0
    # dilation：kernel间距，可用于空间卷积
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    print('----->>> ---->>> vgg backbone : ',len(layers))
    for ii in range(len(layers)):
        print(ii,') vgg layers : ',layers[ii])
    return layers

# imgsize 300  3 个 BasicRFB
extras = {'300': [1024, 'S', 512, 'S', 256]}
#RFBNet 300
def add_extras(size, cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling

    print('add_extras cfg : ',cfg) #[1024, 'S', 512, 'S', 256]
    layers = []
    in_channels = i    # i 对应vgg中conv7的output channels 1024
    flag = False
    # multibox # in_channels = 1024 ,cfg[0] = v = 2014 ,layers += [BasicRFB(in_channels, v, scale = 1.0, visual=2)] ,in_channels = 1024
    # multibox # in_channels = 1024 ,cfg[1] = v = 'S' , layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale = 1.0, visual=2)], in_channels = 'S'
    # in_channels = 'S'  ,cfg[2] = v = 512 , in_channels = 512
    # multibox # in_channels = 512  ,cfg[3] = v = 'S' , layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale = 1.0, visual=2)], in_channels = 'S'
    # in_channels = 'S'  ,cfg[3] = v = 256 , in_channels = 256
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale = 1.0, visual=2)]
            else:
                layers += [BasicRFB(in_channels, v, scale = 1.0, visual=2)]
        in_channels = v
    # 后续 conv 层
    if size ==300:
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1)]# multibox
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1)]# multibox
    else:
        print("Error: Sorry only RFBNet300 are supported!")
        return
    print('add_extras : \n')
    for ii in range(len(layers)):
        print(ii,') add_extras layers : ',layers[ii])
    return layers




def multibox(size, vgg, extra_layers, cfg, num_classes):
    print('/******************  multibox  ****************/')
    print('multibox cfg : ',cfg) # multibox cfg :  [6, 6, 6, 6, 4, 4]
    loc_layers = []
    conf_layers = []
    # 可以发现预测分支是全卷积的，4对应bbox坐标，num_classes对应预测目标类别，如VOC = 21，这个分支对应的是conv4-3
    # bbox
    loc_layers += [nn.Conv2d(512,cfg[0] * 4, kernel_size=3, padding=1)]
    # classify
    conf_layers +=[nn.Conv2d(512,cfg[0] * num_classes, kernel_size=3, padding=1)]

    i = 1
    indicator = 0 # 对应 add_extras 函数返回的新增层数extra_layers
    if size == 300:
        indicator = 3
    else:
        print("Error: Sorry only RFBNet300 are supported!")
        return
    # 对应的参与检测的分支数，cfg内的feature_maps参数
    #第2~6个 feature map 的 grid 的 loc & conf
    for k, v in enumerate(extra_layers):
        if k < indicator or k%2== 0:
            print(' multibox -->> loc_layers : ',k,' default box num :',cfg[i])
            # bbox
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]* 4, kernel_size=3, padding=1)]
            # classify
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]* num_classes, kernel_size=3, padding=1)]
            i +=1
    return vgg, extra_layers, (loc_layers, conf_layers)

# number of boxes per feature map location
mbox = {'300': [6, 6, 6, 6, 4, 4]}
# RFBNet300
# phase : train / test
def build_net(phase, size=300, num_classes=21,debug_ = False):
    return RFBNet(phase, size,\
    *multibox(size, vgg(base[str(size)], 3),add_extras(size, extras[str(size)], 1024),mbox[str(size)], num_classes),\
     num_classes,debug_)

if __name__ == '__main__':
    model_name = 'RFB_Net_vgg'
    # from tensorboardX import SummaryWriter
    model_ = build_net('train', size = 300, num_classes = 21,debug_ =True)
    model_.eval()

    inputs = Variable(torch.rand(5, 3, 300, 300)) # bs,c,h,w   注意图像的尺寸
    model_(inputs)
