import torch
import torch.nn as nn
import numpy as np
from lib.core.config import cfg


class WCEL_Loss(nn.Module):
    """
    Weighted Cross-entropy Loss Function.
    """
    def __init__(self):
        super(WCEL_Loss, self).__init__()
        self.weight = cfg.DATASET.WCE_LOSS_WEIGHT  # list of C
        self.weight /= np.sum(self.weight, 1, keepdims=True)  # (C, C)

    def forward(self, pred_logit, gt_bins, gt, mask, region):
        if region == 'in_range':  # 1 for in range
            mask = 1 - mask
        elif region == 'out_range':  # 1 for out of range
            mask = mask  # [6, 1, 270, 480]

        log_pred = torch.nn.functional.log_softmax(pred_logit, 1)  # [6, 200, 270, 480] | log_softmax -> <0
        log_pred = torch.transpose(log_pred, 0, 1).reshape(log_pred.size(1), -1)
        log_pred = torch.t(log_pred)  # [777600, 200]
        if region != 'all':
            maskPred = mask
            maskPred = torch.transpose(maskPred, 0, 1).reshape(maskPred.size(1), -1)
            maskPred = torch.t(maskPred)  # [777600, 1]
            log_pred = log_pred * maskPred

        classes_range = torch.arange(cfg.MODEL.DECODER_OUTPUT_C, device=gt_bins.device, dtype=gt_bins.dtype)  # [200]
        # print(gt_bins.size())  # [6, 1, 270, 480] for class index
        gt_reshape = gt_bins.reshape(-1, 1)  # [777600, 1]
        one_hot = (gt_reshape == classes_range).to(dtype=torch.float, device=pred_logit.device)  # [777600, 200]
        if region != 'all':
            mask = mask.reshape(-1, 1)  # [6, 1, 270, 480]->[777600, 1]
            one_hot = one_hot * mask

        self.weight = torch.tensor(self.weight, dtype=torch.float32, device=pred_logit.device)  # [200, 200]
        weight = torch.matmul(one_hot, self.weight)  # [777600, 200]
        weight_log_pred = weight * log_pred

        # valid_pixels = torch.sum(gt > 0.).to(dtype=torch.float, device=pred_logit.device)
        valid_pixels = torch.sum(mask > 0.).to(dtype=torch.float, device=pred_logit.device)
        loss = -1 * torch.sum(weight_log_pred) / valid_pixels  # >0

        return loss
