import torch
import torch.nn as nn
import numpy as np
from . import lateral_net as lateral_net
from lib.utils.net_tools import get_func
from lib.models.WCEL_loss import WCEL_Loss
from lib.models.image_transfer import bins_to_depth, kitti_merge_imgs
from lib.core.config import cfg

class MetricDepthModel(nn.Module):
    def __init__(self):
        super(MetricDepthModel, self).__init__()
        self.loss_names = ['Weighted_Cross_Entropy', 'Virtual_Normal']
        self.depth_model = DepthModel()

    def forward(self, data):
        a_real = data['A'].cuda()
        self.b_fake_logit, self.b_fake_softmax = self.depth_model(a_real)
        return {'b_fake_logit': self.b_fake_logit, 'b_fake_softmax': self.b_fake_softmax}

    def inference(self, data):
        with torch.no_grad():
            out = self.forward(data, mode='test')
            if cfg.MODEL.PREDICTION_METHOD == 'classification':
                pred_depth = bins_to_depth(out['b_fake_softmax'])
            elif cfg.MODEL.PREDICTION_METHOD == 'regression':
                pred_depth = torch.nn.functional.sigmoid(out['b_fake_logit'])
            else:
                raise ValueError("Unknown prediction methods")
            out = pred_depth
            return {'b_fake': out}

class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        bottom_up_model = 'lateral_net.lateral_' + cfg.MODEL.ENCODER
        self.encoder_modules = get_func(bottom_up_model)()
        self.decoder_modules = lateral_net.fcn_topdown(cfg.MODEL.ENCODER)

    def forward(self, x):
        lateral_out, encoder_stage_size = self.encoder_modules(x)
        out_logit, out_softmax = self.decoder_modules(lateral_out, encoder_stage_size)
        return out_logit, out_softmax

class ModelLoss(object):
    def __init__(self):
        super(ModelLoss, self).__init__()
        self.weight_cross_entropy_loss = WCEL_Loss()

    def criterion(self, pred_softmax, pred_logit, data, region='all'):
        loss_metric = self.weight_cross_entropy_loss(pred_logit, data['B_bins'], data['B'].cuda(), data['masks'].cuda(), region)

        loss = {}
        loss['metric_loss'] = loss_metric
        loss['total_loss'] = loss['metric_loss']
        return loss

class ModelOptimizer(object):
    def __init__(self, model, parts=['encoder', 'decoder'], type='outer'):
        super(ModelOptimizer, self).__init__()
        encoder_params = []
        encoder_params_names = []
        decoder_params = []
        decoder_params_names = []
        nograd_param_names = []

        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'res' in key:
                    encoder_params.append(value)
                    encoder_params_names.append(key)
                else:
                    decoder_params.append(value)
                    decoder_params_names.append(key)
            else:
                nograd_param_names.append(key)

        if type == 'outer':
            lr_decay = 0.01
            lr_encoder = cfg.TRAIN.BASE_LR * lr_decay
            lr_decoder = cfg.TRAIN.BASE_LR * cfg.TRAIN.SCALE_DECODER_LR * lr_decay
        elif type == 'inner':
            lr_decay = 0.1
            lr_encoder = cfg.TRAIN.BASE_LR * lr_decay
            lr_decoder = cfg.TRAIN.BASE_LR * cfg.TRAIN.SCALE_DECODER_LR * lr_decay
        else:
            lr_encoder = cfg.TRAIN.BASE_LR
            lr_decoder = cfg.TRAIN.BASE_LR * cfg.TRAIN.SCALE_DECODER_LR

        net_params, weight_decay = [], 0.0005
        if 'encoder' in parts:
            net_params.append(
                {'params': encoder_params,
                 'lr': lr_encoder,
                 'weight_decay': weight_decay})
        elif 'decoder' in parts:
            net_params.append(
                {'params': decoder_params,
                 'lr': lr_decoder,
                 'weight_decay': weight_decay}
            )
        else:
            net_params = [
                {'params': encoder_params,
                 'lr': lr_encoder,
                 'weight_decay': weight_decay},
                {'params': decoder_params,
                 'lr': lr_decoder,
                 'weight_decay': weight_decay},
            ]

        self.optimizer = torch.optim.SGD(net_params, momentum=0.9)

    def optim(self, loss):
        self.optimizer.zero_grad()
        loss_all = loss['total_loss']
        loss_all.backward()
        self.optimizer.step()

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss_all = loss['total_loss']
        loss_all.backward()

    def meta_update(self, MetaModel, target_model):
        for p, target_p in zip(MetaModel.parameters(), target_model.parameters()):
            if p.grad is None:
                p.grad = torch.zeros(p.size()).cuda()
            p.grad.data = target_p.data
        self.optimizer.step()

def cal_params(model):
    model_dict = model.state_dict()
    paras = np.sum(p.numel() for p in model.parameters() if p.requires_grad)
    sum = 0

    for key in model_dict.keys():
        print(key)
        if 'layer5' not in key:
            if 'running' not in key:
                print(key)
                ss = model_dict[key].size()
                temp = 1
                for s in ss:
                    temp = temp * s
                print(temp)
                sum = sum + temp
    print(sum)
    print(paras)
