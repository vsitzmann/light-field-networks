import torch.nn as nn


def image_loss(model_out, gt, mask=None):
    gt_rgb = gt['rgb']
    return nn.MSELoss()(gt_rgb, model_out['rgb']) * 200


class LFLoss():
    def __init__(self, l2_weight=1, reg_weight=1e2):
        self.l2_weight = l2_weight
        self.reg_weight = reg_weight

    def __call__(self, model_out, gt, model=None, val=False):
        loss_dict = {}
        loss_dict['img_loss'] = image_loss(model_out, gt)
        loss_dict['reg'] = (model_out['z']**2).mean() * self.reg_weight
        return loss_dict, {}
    
    
class SRNLoss():
    def __init__(self, l2_weight=1, reg_weight=1e2):
        self.l2_weight = l2_weight
        self.reg_weight = reg_weight

    def __call__(self, model_out, gt, model=None, val=False):
        loss_dict = {}
        loss_dict['img_loss'] = image_loss(model_out, gt)
        loss_dict['reg'] = (model_out['z']**2).mean() * self.reg_weight
        loss_dict['depth_reg'] = (torch.min(model_out['depth'], torch.zeros_like(model_out['depth'])) ** 2) * 10000
        return loss_dict, {}


