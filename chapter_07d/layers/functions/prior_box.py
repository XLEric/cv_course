import torch
from math import sqrt as sqrt
from itertools import product as product


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """
    # multibox cfg :  [6, 6, 6, 6, 4, 4]
    def __init__(self, cfg, debug_ = False):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim'] # VOC300 - 300
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios']) # VOC300 - [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.variance = cfg['variance'] or [0.1] # VOC300 - [0.1, 0.2]
        self.feature_maps = cfg['feature_maps'] # VOC300 - [38, 19, 10, 5, 3, 1]
        self.min_sizes = cfg['min_sizes'] # VOC300 - [30, 60, 111, 162, 213, 264]
        self.max_sizes = cfg['max_sizes'] # VOC300 - [60, 111, 162, 213, 264, 315]
        self.steps = cfg['steps'] # VOC300 - [8, 16, 32, 64, 100, 300]
        self.aspect_ratios = cfg['aspect_ratios'] # VOC300 - [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.clip = cfg['clip'] # VOC300 - True
        self.debug_ = debug_
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):#VOC300 - 遍历6特征图，每个特征图分别生成默认框
            if self.debug_:
                print('feature_maps : ',k)

            for i, j in product(range(f), repeat=2):#每个特征图遍历所有 anchor 坐标 ：i,j 是 grid 的格子id
                """
                将特征图的anchor 中心坐标对应回原图坐标，然后缩放成0-1的相对距离
                原始公式应该为cx = (j+0.5) * step /min_dim，拆分成两步计算
                """
                # step 1
                f_k = self.image_size / self.steps[k]#每一个 feature map - step 代表的像素值
                # step 2
                cx = (j + 0.5) / f_k  # 针对 不同的 feature map ，全图 归一化的中心坐标
                cy = (i + 0.5) / f_k
                """ ratio为1 的 anchor """
                # anchor size : min_sizes
                s_k = self.min_sizes[k]/self.image_size# anchor 归一化
                mean += [cx, cy, s_k, s_k]
                if self.debug_:
                    print('(%s,%s) : (%s,%s,%s,%s)'%(i,j,cx, cy, s_k, s_k))

                # aspect_ratio: 1
                # self.max_sizes[k]/self.image_size 为 最大边的归一化
                # size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))# anchor 归一化
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios ： 除了 1:1比例 anchor ，其它比例的 anchor
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]# anchor 归一化
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # back to torch land
        """默认框转化为n行4列的标准形式，每行一个默认框[x,y,w,h]"""
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            '''torch.clamp(input,min,max,out=None)将输入input张量每个元素的范围限制到区间[min,max]'''
            output.clamp_(max=1, min=0)
        # print('PriorBox \n',output)
        return output
