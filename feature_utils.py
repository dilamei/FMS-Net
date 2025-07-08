import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def tensor_to_heatmap(tensor_2d):
    tensor_2d = tensor_2d.cpu().numpy()
    tensor_2d = (tensor_2d * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(tensor_2d, cv2.COLORMAP_JET)
    return heatmap

def overlay_heatmap_on_image(heatmap, image, alpha=0.6):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)

def save_feature_heatmaps(tensor, save_dir, original_image=None, prefix='fmap', max_channels=4):
    os.makedirs(save_dir, exist_ok=True)
    tensor = tensor[:max_channels]

    for i, t in enumerate(tensor):
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-5)
        heatmap = tensor_to_heatmap(t_norm)

        filename = os.path.join(save_dir, f'{prefix}_channel{i}.png')
        cv2.imwrite(filename, heatmap)

        if original_image is not None:
            overlay = overlay_heatmap_on_image(heatmap, original_image)
            cv2.imwrite(os.path.join(save_dir, f'{prefix}_overlay{i}.png'), overlay)
