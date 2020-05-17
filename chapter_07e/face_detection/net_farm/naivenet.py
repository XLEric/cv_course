"""
NaiveNet is in a style which is a deliberately very simple
convolutional neural network backbone aiming at deploying
on all platforms easily, but can also get balance on accuracy
and efficient extremely at the same time. The entire backbone
only consists of conv 3×3, conv 1×1, ReLU and residual connection.
"""

import torch
import torch.nn as nn


# num_filters_list = [32, 64, 128, 256]
num_filters_list = [16, 32, 64, 128]

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=True)


def conv1x1(in_channels, out_channels):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, 1, bias=True)


class Resv1Block(nn.Module):
    """ResNet v1 block without bn"""
    def __init__(self, inplanes, planes, stride=1):
        super(Resv1Block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, stride)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out += x
        out = self.relu2(out)
        return out


class Resv2Block(nn.Module):
    """ResNet v2 block without bn"""
    def __init__(self, inplanes, planes, stride=1, is_branch=False):
        super(Resv2Block, self).__init__()
        self.is_branch = is_branch
        self.relu1 = nn.ReLU()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, stride)

    def forward(self, x):
        out_branch = self.relu1(x)
        out = self.conv1(out_branch)
        out = self.relu2(out)
        out = self.conv2(out)
        out += x
        if self.is_branch:
            return out, out_branch
        else:
            return out


class BranchNet(nn.Module):
    """
    The branch of NaiveNet is the network output and
    only consists of conv 1×1 and ReLU.
    """
    def __init__(self, inplanes, planes):
        super(BranchNet, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.conv2_score = conv1x1(planes, planes)
        self.conv3_score = conv1x1(planes, 2)
        self.conv2_bbox = conv1x1(planes, planes)
        self.conv3_bbox = conv1x1(planes, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out_score = self.conv2_score(out)
        out_score = self.relu(out_score)
        out_score = self.conv3_score(out_score)

        out_bbox = self.conv2_bbox(out)
        out_bbox = self.relu(out_bbox)
        out_bbox = self.conv3_bbox(out_bbox)

        return out_score, out_bbox


class NaiveNet(nn.Module):
    """NaiveNet for Fast Single Class Object Detection.
    The entire backbone and branches only consists of conv 3×3,
    conv 1×1, ReLU and residual connection.
    """
    def __init__(self, arch, block, layers):
        super(NaiveNet, self).__init__()
        self.arch = arch
        self.block = block
        if self.arch == 'naivenet25':
            if self.block == Resv2Block:
                self.conv1 = conv3x3(3, num_filters_list[1], stride=2, padding=0)
                self.relu1 = nn.ReLU()
                self.stage1_1 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[1], layers[0] - 1, stride=2)
                self.stage1_2_branch1 = nn.Sequential(self.block(num_filters_list[1], num_filters_list[1], stride=1, is_branch=True))
                self.branch1 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage1_3_branch2 = nn.Sequential(nn.ReLU())
                self.branch2 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage2_1 = nn.Sequential(conv3x3(num_filters_list[1], num_filters_list[1], stride=2, padding=0),
                                              Resv2Block(num_filters_list[1], num_filters_list[1], stride=1, is_branch=False))
                self.stage2_2_branch3 = nn.Sequential(Resv2Block(num_filters_list[1], num_filters_list[1], stride=1, is_branch=True))
                self.branch3 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage2_3_branch4 = nn.Sequential(nn.ReLU())
                self.branch4 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage3_1 = nn.Sequential(conv3x3(num_filters_list[1], num_filters_list[2], stride=2, padding=0),
                                              Resv2Block(num_filters_list[2], num_filters_list[2], stride=1, is_branch=False))
                self.stage3_2_branch5 = nn.Sequential(nn.ReLU())
                self.branch5 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
                self.stage4_1 = nn.Sequential(conv3x3(num_filters_list[2], num_filters_list[2], stride=2, padding=0),
                                              Resv2Block(num_filters_list[2], num_filters_list[2], stride=1, is_branch=False))
                self.stage4_2_branch6 = nn.Sequential(Resv2Block(num_filters_list[2], num_filters_list[2], stride=1, is_branch=True))
                self.branch6 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
                self.stage4_3_branch7 = nn.Sequential(Resv2Block(num_filters_list[2], num_filters_list[2], stride=1, is_branch=True))
                self.branch7 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
                self.stage4_4_branch8 = nn.Sequential(nn.ReLU())
                self.branch8 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
            elif self.block == Resv1Block:
                self.conv1 = conv3x3(3, num_filters_list[1], stride=2, padding=0)
                self.relu1 = nn.ReLU()
                self.stage1_1 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[1], layers[0] - 1, stride=2)
                self.branch1 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage1_2 = nn.Sequential(self.block(num_filters_list[1], num_filters_list[1], stride=1))
                self.branch2 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage2_1 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[1], layers[1] - 1, stride=2)
                self.branch3 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage2_2 = nn.Sequential(self.block(num_filters_list[1], num_filters_list[1], stride=1))
                self.branch4 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage3_1 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[2], layers[2], stride=2)
                self.branch5 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
                self.stage4_1 = self._make_layer(self.arch, self.block, num_filters_list[2], num_filters_list[2], layers[3] - 2, stride=2)
                self.branch6 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
                self.stage4_2 = nn.Sequential(self.block(num_filters_list[2], num_filters_list[2], stride=1))
                self.branch7 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
                self.stage4_3 = nn.Sequential(self.block(num_filters_list[2], num_filters_list[2], stride=1))
                self.branch8 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
            else:
                raise TypeError('Unsupported ResNet Block Version.')
        elif self.arch == 'naivenet20':
            self.conv1 = conv3x3(3, num_filters_list[1], stride=2, padding=0)
            self.relu1 = nn.ReLU()
            self.layer1 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[1], layers[0], stride=2)
            self.branch1 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
            self.layer2 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[1], layers[1], stride=2)
            self.branch2 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
            self.layer3 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[1], layers[2], stride=2)
            self.branch3 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
            self.layer4 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[2], layers[3], stride=2)
            self.branch4 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
            self.layer5 = self._make_layer(self.arch, self.block, num_filters_list[2], num_filters_list[2], layers[4], stride=2)
            self.branch5 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))

        else:
            raise TypeError('Unsupported NaiveNet Version.')


    def _make_layer(self, arch, block, inplanes, planes, blocks, stride=2):
        layers = []
        if self.arch == 'naivenet25':
            if block == Resv2Block:
                layers.append(conv3x3(inplanes, planes, stride=stride, padding=0))
                for _ in range(blocks):
                    layers.append(block(planes, planes, stride=1))
            elif block == Resv1Block:
                layers.append(conv3x3(inplanes, planes, stride=stride, padding=0))
                layers.append(nn.ReLU())
                for _ in range(blocks):
                    layers.append(block(planes, planes, stride=1))
            else:
                raise TypeError('Unsupported ResNet Block Version.')
        elif self.arch == 'naivenet20':
            if block == Resv2Block:
                layers.append(conv3x3(inplanes, planes, stride=stride, padding=0))
                for _ in range(blocks):
                    layers.append(block(planes, planes, stride=1))
                layers.append(nn.ReLU())
            elif block == Resv1Block:
                layers.append(conv3x3(inplanes, planes, stride=stride, padding=0))
                layers.append(nn.ReLU())
                for _ in range(blocks):
                    layers.append(block(planes, planes, stride=1))
            else:
                raise TypeError('Unsupported ResNet Block Version.')
        else:
            raise TypeError('Unsupported NaiveNet Version.')

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.arch == 'naivenet25':
            if self.block == Resv2Block:
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.stage1_1(x)
                x, b1 = self.stage1_2_branch1(x)
                score1, bbox1 = self.branch1(b1)
                x = b2 = self.stage1_3_branch2(x)
                score2, bbox2 = self.branch2(b2)
                x = self.stage2_1(x)
                x, b3 = self.stage2_2_branch3(x)
                score3, bbox3 = self.branch3(b3)
                x = b4 = self.stage2_3_branch4(x)
                score4, bbox4 = self.branch4(b4)
                x = self.stage3_1(x)
                x = b5 = self.stage3_2_branch5(x)
                score5, bbox5 = self.branch5(b5)
                x = self.stage4_1(x)
                x, b6 = self.stage4_2_branch6(x)
                score6, bbox6 = self.branch6(b6)
                x, b7 = self.stage4_3_branch7(x)
                score7, bbox7 = self.branch7(b7)
                x = b8 = self.stage4_4_branch8(x)
                score8, bbox8 = self.branch8(b8)

            if self.block == Resv1Block:
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.stage1_1(x)
                score1, bbox1 = self.branch1(x)
                x = self.stage1_2(x)
                score2, bbox2 = self.branch2(x)
                x = self.stage2_1(x)
                score3, bbox3 = self.branch3(x)
                x = self.stage2_2(x)
                score4, bbox4 = self.branch4(x)
                x = self.stage3_1(x)
                score5, bbox5 = self.branch5(x)
                x = self.stage4_1(x)
                score6, bbox6 = self.branch6(x)
                x = self.stage4_2(x)
                score7, bbox7 = self.branch7(x)
                x = self.stage4_3(x)
                score8, bbox8 = self.branch8(x)
                outs = [score1, bbox1, score2, bbox2, score3, bbox3, score4, bbox4, score5, bbox5, score6, bbox6, score7, bbox7, score8, bbox8]
            return outs

            # uncomment to display with torchviz
            # return score1, bbox1, score2, bbox2, score3, bbox3, score4, bbox4, score5, bbox5, score6, bbox6, score7, bbox7, score8, bbox8

        if self.arch == 'naivenet20':
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.layer1(x)
            score1, bbox1 = self.branch1(x)
            x = self.layer2(x)
            score2, bbox2 = self.branch2(x)
            x = self.layer3(x)
            score3, bbox3 = self.branch3(x)
            x = self.layer4(x)
            score4, bbox4 = self.branch4(x)
            x = self.layer5(x)
            score5, bbox5 = self.branch5(x)
            outs = [score1, bbox1, score2, bbox2, score3, bbox3, score4, bbox4, score5, bbox5]
            return outs

            # uncomment to display with torchviz
            # return score1, bbox1, score2, bbox2, score3, bbox3, score4, bbox4, score5, bbox5



def get_naivenet(arch, block, layers):
    model = NaiveNet(arch, block, layers)
    # model = BackboneNet(arch, block, layers)
    return model


def naivenet25():
    r"""NaiveNet-25 model from
    `"LFFD: A Light and Fast Face Detector for Edge Devices" <https://arxiv.org/pdf/1904.10633.pdf>`_
    It corresponds to the network structure built by `symbol_10_560_25L_8scales_v1.py` of mxnet version.
    """
    return get_naivenet('naivenet25', Resv2Block, [4, 2, 1, 3])
    # return get_naivenet('naivenet25', Resv1Block, [4, 2, 1, 3])


def naivenet20():
    r"""NaiveNet-20 model from
    `"LFFD: A Light and Fast Face Detector for Edge Devices" <https://arxiv.org/pdf/1904.10633.pdf>`_
    It corresponds to the network structure built by `symbol_10_320_20L_5scales_v2.py` of mxnet version.
    """
    return get_naivenet('naivenet20', Resv2Block, [3, 1, 1, 1, 1])
    # return get_naivenet('naivenet20', Resv1Block, [3, 1, 1, 1, 1])


if __name__ == '__main__':
    import os
    from torch.autograd import Variable
    from torchsummary import summary
    from torchviz import make_dot
    import tensorwatch

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    x_image = Variable(torch.randn(8, 3, 640, 640))

    # net = naivenet25()
    net = naivenet20()
    print(net)
    y = net(x_image)
    # print(y)

    summary(net.to('cuda'), (3, 640, 640))

    # tensorwatch.draw_model(net, [1, 3, 640, 640])

    """
    If you want to show with torchviz,
    you need to modify the return format of the network.
    """
    # vis_graph = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x_image)]))

    # # vis_graph.format = 'png'
    # # vis_graph.format = 'pdf'
    # vis_graph.format = 'svg'
    # # vis_graph.render('naivenet25_resv2.gv')
    # vis_graph.render('naivenet20_resv2.gv')
    # # vis_graph.render('naivenet25_resv1.gv')
    # # vis_graph.render('naivenet20_resv1.gv')
    # vis_graph.view()
