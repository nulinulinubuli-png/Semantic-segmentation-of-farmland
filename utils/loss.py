import torch
import torch.nn as nn
import torch.nn.functional as F
class SegmentationLosses(object):
    def __init__(self, weight=None, reduction='mean', batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal','dice']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dice':
            return self.Diceloss
        elif mode == 'bce':
            return self.BCEWithLogitsLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        # print(logit.size())
        # print(target.size())
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                     reduction=self.reduction)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def structure_loss(self,pred, mask):

        # BCE loss
        # n, c, h, w = pred.size()

        k = nn.Softmax2d()
        weit = torch.abs(pred - mask)
        weit = k(weit)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # IOU loss
        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()/4


    def BCEWithLogitsLoss(self, logit, target):
        # print("BCE")
        n,  h, w = target.size()
        logit = logit.view(n, 256, 256)

        # lossiou = self.structure_loss(logit,target)
        criterion = nn.BCEWithLogitsLoss(weight=self.weight, reduction=self.reduction)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target)

        if self.batch_average:
            loss /= n

        return loss


    def binary_cross_entropy_loss_with_logits(self, logit, target):
        n, c, h, w = logit.shape
        target = target.view(n, -1)  # Flatten y_actual to match the shape of logits
        logit = logit.view(n, c, -1).permute(0, 2, 1)  # Reshape logits for cross-entropy calculation
        logit = logit.contiguous().view(-1, c)

        loss = -torch.sum(
            target * torch.log(torch.sigmoid(logit)) + (1 - target) * torch.log(1 - torch.sigmoid(logit))) / n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss
    def Diceloss(self, logit, target, beta=1, smooth = 1e-5):
        n, c, h, w = logit.size()
        nt, ht, wt, ct = target.size()
        if h != ht and w != wt:
            logit = F.interpolate(logit, size=(ht, wt), mode="bilinear", align_corners=True)

        temp_inputs = torch.softmax(logit.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
        temp_target = target.view(n, -1, ct)
        tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
        fp = torch.sum(temp_inputs, axis=[0,1]) - tp
        fn = torch.sum(temp_target[...,:-1], axis=[0,1]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        dice_loss = 1 - torch.mean(score)
        return dice_loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




