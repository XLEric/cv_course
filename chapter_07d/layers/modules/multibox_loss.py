import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
GPU = False
if torch.cuda.is_available():
    GPU = True


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    # num_classes, 0.5, True, 0, True, 3, 0.5, False
    def __init__(self, num_classes,overlap_thresh,prior_for_matching,bkg_label,neg_mining,neg_pos,neg_overlap,encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching  = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1,0.2]# for loc

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0)) #11620   all feature_maps grid default bboxes number
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        """
        for循环中的代码是为了将输入的target改造成网络的学习目标,也就是计算损失时的target最终得到的是loc_t和conf_t
        注:对于Tensor来说,在子函数中修改其值,原有的值也会跟着改变,因此match函数无返回值
        """
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:,:-1].data# 真实loc
            labels = targets[idx][:,-1].data# 真实label
            defaults = priors.data
            match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)# gt 和 default boxes
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)

        """
        conf_t > 0等价于torch.gt(conf_t,0)或者conf_t.gt(0)
        返回和conf_t同形状的Tensor,符合条件的为1,否则为0
        """
        pos = conf_t > 0 #忽略背景,pos是mask

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)# predictions
        loc_t = loc_t[pos_idx].view(-1,4)# encoded offsets to learn
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        """
        conf_data Shape:[batch,num_priors,num_classes]
        batch_conf Shape:[batch*num_priors,num_classes]
        因为pytorch中cross_entropy的input要求为[N,C]的2-d Tensor
        """
        batch_conf = conf_data.view(-1,self.num_classes)#predictions
        """
        conf_t的shape为[batch,num_priors],其中选中的正样本为相应的类别，未选中的为0
        Tensor.gather(dim,index)在dim维度上，按照index = 1 。此处就是计算cross_entropy的x[class]项
        loss(x,class) = −x[class]+log(∑jexp(x[j]))
        """
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))#为了筛选负样本

        # Hard Negative Mining
        """
        先将正样本loss置为0，然后对loss排序(每张图片内部挑选)之后，取前self.negpos_ratio*num_pos个负样本的loss
        """
        loss_c[pos.view(-1,1)] = 0 # filter out pos boxes,为了选择负样本。
        """
        下一步loss_c shape转变为[batch,num_priors]
        下面这种挑选前n个数的操作
        """
        loss_c = loss_c.view(num, -1)

        a_,loss_idx = loss_c.sort(1, descending=True)
        # print('loss_idx : ',loss_idx)
        # print('loss_idx : ',loss_idx.size(),a_.size())
        # print('a_ : ',a_)
        b_,idx_rank = loss_idx.sort(1)
        # print('idx_rank : ',idx_rank)
        # print('b_ : ',b_)
        # print('loss_idx : ',idx_rank.size(),b_.size())
        num_pos = pos.long().sum(1,keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)# 夹紧范围，限制 num_neg的范围
        neg = idx_rank < num_neg.expand_as(idx_rank)# 负样本 index 掩码

        # print(neg.size(),' neg',neg)
        # print('neg sum : ',neg.sum())
        # print('num_pos : ',num_pos.sum(),'\n',num_pos)

        # Confidence Loss Including Positive and Negative Examples
        """
        上面几步的操作就是为获得pos_idx和neg_idx
        conf_data 的shape为[batch,num_priors,num_classes]
        """
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        """
        (pos_idx+neg_idx).gt(0)的原因个人猜测可能是因为挑选的正样本和负样本可能会重复，因此将大于1的数变成1.
        """
        # gt举例：torch.gt(x,1)# tensor 大于 1
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = max(num_pos.data.sum().float(), 1)
        loss_l/=N
        loss_c/=N
        return loss_l,loss_c
