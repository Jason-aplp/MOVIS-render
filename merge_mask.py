import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import json

parser = argparse.ArgumentParser(description="Parameter for merging masks.")
parser.add_argument('--path', type=str, default='example_mix.json')
args = parser.parse_args()

with open(args.path, 'r') as f:
    folder_pth = json.load(f)
for i in tqdm(folder_pth):
    for k in range(12):
        mask_pth = os.path.join(i, f"mask_{k:03d}")
        if not os.path.exists(os.path.join(mask_pth, 'mixed_mask.png')):
            img_lst = os.listdir(mask_pth)
            img_lst.sort()
            cur_num = 2
            empty = np.ones((512, 512))
            for img in img_lst:
                if (int(img.split('_')[-3]) == 0):
                    tmp_img = cv2.imread(os.path.join(mask_pth, img), cv2.IMREAD_GRAYSCALE)
                    empty[np.where(tmp_img != 0)] = cur_num
                    cur_num += 1
            cv2.imwrite(os.path.join(mask_pth, 'mixed_mask.png'), empty)