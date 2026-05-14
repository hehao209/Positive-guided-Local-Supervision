import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import cv2
import numpy as np
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)

        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_pred, y_true):
        a =  self.bce_loss(y_pred, y_true)
        b =  self.soft_dice_loss(y_pred, y_true)
        return a + b



class local_dice_bce_loss(nn.Module):
    def __init__(self, batch=True, num_regions=16, crop_sizes=[256, 128, 64, 32]):
        super().__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        self.num_regions = num_regions
        self.crop_sizes = crop_sizes  

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1e-6  
        intersection = torch.sum(y_true * y_pred, dim=[1,2,3])
        union = torch.sum(y_true, dim=[1,2,3]) + torch.sum(y_pred, dim=[1,2,3])
        return (2. * intersection + smooth) / (union + smooth)

    def soft_dice_loss(self, y_true, y_pred):
        return 1 - self.soft_dice_coeff(y_true, y_pred).mean()

    def _get_positive_coords(self, mask):
        B, _, H, W = mask.shape
        coords = torch.zeros((B, self.num_regions, 2), dtype=torch.long, device=mask.device)
        
        for b in range(B):
            pos_mask = mask[b].squeeze() > 0.5
            if pos_mask.any():
                y_idx, x_idx = torch.where(pos_mask)
                if len(y_idx) == 0:
                    # 处理空坐标
                    coords[b] = torch.tensor([[H//2, W//2]], device=mask.device).repeat(self.num_regions, 1)
                    continue
                    
                indices = torch.randint(0, len(y_idx), (self.num_regions,))
                coords[b] = torch.stack((y_idx[indices], x_idx[indices]), dim=1)
            else:
                # 生成中心坐标
                center = torch.tensor([H//2, W//2], device=mask.device)
                coords[b] = center.repeat(self.num_regions, 1)
                
        return coords

    def _smart_crop(self, tensor, coords, crop_size):
        B, K = coords.shape[:2]
        S = crop_size
        crops = torch.zeros((B, K, tensor.shape[1], S, S), device=tensor.device)
        
        for b in range(B):
            for k in range(K):
                y, x = coords[b, k]
                y_start = max(0, y - S//2)
                x_start = max(0, x - S//2)
                y_end = min(tensor.shape[2], y_start + S)
                x_end = min(tensor.shape[3], x_start + S)
                
                crop = tensor[b:b+1, :, y_start:y_end, x_start:x_end]
                _, _, h, w = crop.shape
                if h < S or w < S:
                    # 填充不足部分
                    pad = nn.ConstantPad2d((0, S-w, 0, S-h), 0)
                    crop = pad(crop)
                crops[b, k] = crop.squeeze(0)
        return crops

    def __call__(self, y_pred, y_true):
        crop_size = np.random.choice(self.crop_sizes)
        coords = self._get_positive_coords(y_true)
        true_crops = self._smart_crop(y_true, coords, crop_size)
        pred_crops = self._smart_crop(y_pred, coords, crop_size)
        
        loss = 0
        for k in range(self.num_regions):
            bce = self.bce_loss(pred_crops[:,k], true_crops[:,k])
            dice = self.soft_dice_loss(pred_crops[:,k], true_crops[:,k])
            loss += (bce + dice)
            
        return loss / self.num_regions

