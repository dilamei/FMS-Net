import numpy as np

def cal_mae(pred, gt):
    """Calculate Mean Absolute Error (MAE)"""
    return np.mean(np.abs(pred - gt))

def cal_fm(pred, gt, beta2=0.3):
    """Calculate F-measure"""
    # Ensure both arrays are 2D
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    
    # Normalize
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
    
    # Threshold predictions
    thresh = 2 * pred.mean()
    binary = (pred > thresh).astype(np.float32)
    
    # Calculate TP, precision and recall
    tp = (binary * gt).sum()
    precision = tp / (binary.sum() + 1e-8)
    recall = tp / (gt.sum() + 1e-8)
    
    # Calculate F-measure
    fm = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
    return fm

def cal_sm(pred, gt, alpha=0.5):
    """Calculate Structure-measure (S-measure)"""
    # Ensure both arrays are 2D
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    
    # Normalize
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
    
    y = gt.mean()
    
    if y == 0:
        return 1 - pred.mean()
    elif y == 1:
        return pred.mean()
    else:
        # Generate binary maps
        gt_binary = (gt >= 0.5)
        pred_binary = (pred >= pred.mean())
        
        # Calculate foreground values
        fg_sum = pred[gt_binary].sum()
        fg_cnt = gt_binary.sum()
        fg_mean = fg_sum / (fg_cnt + 1e-8)
        fg_std = np.sqrt(((pred[gt_binary] - fg_mean) ** 2).sum() / (fg_cnt + 1e-8))
        
        # Calculate background values
        bg_sum = pred[~gt_binary].sum()
        bg_cnt = (~gt_binary).sum()
        bg_mean = bg_sum / (bg_cnt + 1e-8)
        bg_std = np.sqrt(((pred[~gt_binary] - bg_mean) ** 2).sum() / (bg_cnt + 1e-8))
        
        # Object structure similarity
        obj_sim = 2 * fg_mean / (fg_mean * fg_mean + 1 + fg_std + 1e-8)
        
        # Background structure similarity
        bg_sim = 2 * (1 - bg_mean) / ((1 - bg_mean) * (1 - bg_mean) + 1 + bg_std + 1e-8)
        
        # Combine similarities
        w = gt.sum() / (gt.size + 1e-8)
        sm = w * obj_sim + (1 - w) * bg_sim
        
        return sm

def cal_em(pred, gt):
    """Calculate Enhanced-measure (E-measure)"""
    # Ensure both arrays are 2D
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    
    # Normalize
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
    
    # Generate binary maps
    gt_binary = (gt >= 0.5)
    
    # Calculate foreground and background means
    fg_mean = pred[gt_binary].mean() if gt_binary.any() else 0
    bg_mean = pred[~gt_binary].mean() if (~gt_binary).any() else 0
    
    # Alignment matrix
    align = 1 - np.abs(pred - gt)
    
    # Enhanced alignment matrix
    enhanced = align.copy()
    enhanced[gt_binary] *= (1 - fg_mean)
    enhanced[~gt_binary] *= (1 - bg_mean)
    
    return enhanced.mean()