from __future__ import print_function, division, absolute_import
import torch
import numpy as np
import random
from sklearn.metrics import average_precision_score, accuracy_score, roc_curve, auc

__all__ = ["data_prefetcher", "data_prefetcher_two", "cal_fam", "cal_normfam", "setup_seed", "l2_norm", "calRes"]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class data_prefetcher():
    def __init__(self, loader):
        self.stream = torch.cuda.Stream()
        self.loader = iter(loader)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).long()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class data_prefetcher_two():
    def __init__(self, loader1, loader2):
        self.stream = torch.cuda.Stream()
        self.loader1 = iter(loader1)
        self.loader2 = iter(loader2)
        self.preload()

    def preload(self):
        try:
            tmp_input1, tmp_target1 = next(self.loader1)
            tmp_input2, tmp_target2 = next(self.loader2)
            self.next_input, self.next_target = torch.cat((tmp_input1, tmp_input2)), torch.cat((tmp_target1, tmp_target2))

        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).long()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm+1e-8)
    return output


def cal_fam(model, inputs):
    model.zero_grad()
    inputs = inputs.detach().clone()
    inputs.requires_grad_()
    output = model(inputs)

    target = output[:, 1]-output[:, 0]
    target.backward(torch.ones(target.shape).cuda())
    fam = torch.abs(inputs.grad)
    fam = torch.max(fam, dim=1, keepdim=True)[0]
    return fam


def cal_normfam(model, inputs):
    fam = cal_fam(model, inputs)
    _, x, y = fam[0].shape
    fam = torch.nn.functional.interpolate(fam, (int(y/2), int(x/2)), mode='bilinear', align_corners=False)
    fam = torch.nn.functional.interpolate(fam, (y, x), mode='bilinear', align_corners=False)
    for i in range(len(fam)):
        fam[i] -= torch.min(fam[i])
        fam[i] /= torch.max(fam[i])
    return fam


def calRes(y_true_all, y_pred_all):
    y_true_all, y_pred_all = np.array(
        y_true_all.cpu()), np.array(y_pred_all.cpu())

    fprs, tprs, ths = roc_curve(
        y_true_all, y_pred_all, pos_label=1, drop_intermediate=False)

    acc = accuracy_score(y_true_all, np.where(y_pred_all >= 0.5, 1, 0))*100.

    ind = 0
    for fpr in fprs:
        if fpr > 1e-2:
            break
        ind += 1
    TPR_2 = tprs[ind-1]

    ind = 0
    for fpr in fprs:
        if fpr > 1e-3:
            break
        ind += 1
    TPR_3 = tprs[ind-1]

    ind = 0
    for fpr in fprs:
        if fpr > 1e-4:
            break
        ind += 1
    TPR_4 = tprs[ind-1]

    ap = average_precision_score(y_true_all, y_pred_all)
    return ap, acc, auc(fprs, tprs), TPR_2, TPR_3, TPR_4
