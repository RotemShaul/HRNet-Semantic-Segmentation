# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset

class MidAir2c(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=2,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 mean_disp=41.42,
                 std_disp=31.03):

        super(MidAir2c, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.mean_disp = mean_disp
        self.std_disp = std_disp

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.class_weights = torch.FloatTensor([18.0032, 0.9969]).cuda()

        self.multi_scale = multi_scale
        self.flip = flip
        self.center_crop_test = center_crop_test
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path = item
                name = image_path
                image_path = 'color_left/trajectory_5000/' + image_path[0: len(image_path) - 1]
                label_path = 'segmentation/trajectory_5000/' + image_path[0: len(image_path) - 5] + 'PNG'
                disparity_path = 'stereo_disparity/trajectory_5000/' + image_path[0: len(image_path) - 5] + 'PNG'
                #print("Calculated image path and disp path {} {}".format(image_path, disparity_path))
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "disparity": disparity_path,
                    "name": name,
                    "weight": 1
                })
        return files
        
    def convert_label(self, label, inverse=False):
        return label

    def __getitem__(self, index):

        item = self.files[index]
        name = item["name"]
        image_path = item["img"]
        label_path = item["label"]
        disparity_path = item["disparity"]

        rgb_image = Image.open(os.path.join(self.root,'midair',image_path))
        seg_label = Image.open(os.path.join(self.root,'midair',label_path))
        disp_img = Image.open(os.path.join(self.root,'midair',disparity_path))

        size = rgb_image.shape

        rgb_image = np.asarray(rgb_image, np.float32)
        seg_label = np.asarray(seg_label, np.int64)

        # seg_label[seg_label == 0] = 14 #Give the background class label
        seg_label[seg_label == 2] = -1  # Tree
        seg_label[seg_label >= 0] = 0
        seg_label[seg_label == -1] = 1 #Tree

        disp_img = np.asarray(disp_img, np.uint16)
        disp_img.dtype = np.float16
        disp_img = disp_img.astype(np.float32)
        label = np.asarray(seg_label, dtype=np.int32)

        ############

        return rgb_image.copy(), label.copy(), np.array(size), name, disp_img.copy()

    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]
            preds = F.upsample(preds, (ori_height, ori_width), 
                                   mode='bilinear')
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))



