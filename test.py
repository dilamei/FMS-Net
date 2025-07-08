import torch
import torch.nn.functional as F
import numpy as np
import pdb, os, argparse
import imageio
import time
import cv2

from model.PVTWithBERT import PVTwithBERT
from data import test_dataset

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt = parser.parse_args()

dataset_path = './dataset/test_dataset/'

model = PVTwithBERT()
model.cuda()
model.eval()

# test_datasets = ['EORSSD']
test_datasets = ['ORSSD']
# test_datasets = ['ORSI-4199']

for epoch in range(70, 101, 1):
    model_path = f'./results/ORSSD/checkpoint_epoch_{epoch}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    for dataset in test_datasets:
        save_path = f'./sal_map/ORSSD/epoch_{epoch}/{dataset}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image_root = '/data/dlm/dataset/ORSSD/Image-test/'
        gt_root = '/data/dlm/dataset/ORSSD/GT-test/'
        text_root = '/data/dlm/dataset/ORSSD/test_descriptions.json'
        
        #image_root = '/data/dlm/dataset/EORSSD/Image-test/'
        #gt_root = '/data/dlm/dataset/EORSSD/GT-test/'
        #text_root = '/data/dlm/dataset/EORSSD/test_descriptions.json'
        
        # image_root = '/data/dlm/dataset/ors-4199/testset/images/'
        # gt_root = '/data/dlm/dataset/ors-4199/testset/gt/'

        print(f'Testing epoch {epoch} on dataset {dataset}')

        test_loader = test_dataset(image_root, gt_root, text_root, opt.testsize)
        time_sum = 0
        
        for i in range(test_loader.size):
            image, gt, text, name = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            image = image.cuda()
            text = text.cuda()
            time_start = time.time()
            res, s2, s3, s4 = model(image, text)

            time_end = time.time()
            time_sum = time_sum+(time_end-time_start)
            res = F.interpolate(res, size=gt.shape[2:], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res *  255.0).astype(np.uint8)
            imageio.imsave(save_path+name, res)

            if i == test_loader.size - 1:
                print(f'Epoch {epoch}: Running time: {time_sum / test_loader.size:.5f} seconds')
                print(f'Epoch {epoch}: Average speed: {test_loader.size / time_sum:.4f} fps')