# -*- coding:utf-8 -*-
#@Time : 2022/9/27 11:31
#@Author: sunrise
#@File : ims_pascal.py
#@Todo : 双端注意力

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
import torch
import numpy as np


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 args,
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        #  这里我们的数据集应该是15分类，加上背景总共16分类

        super().__init__()
        self.args = args
        self._base_dir = Path.db_root_dir(self.args.dataset)
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args
        self.NUM_CLASSES = self.args.num_class

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.gimages = []
        self.categories = []
        self.firename = []
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + "_" + self.args.suffix[0] + ".tif")
                # print(_image)
                _g_image = os.path.join(self._image_dir, line + "_" + self.args.suffix[0] + ".tif")

                # print(_g_image)
                _cat = os.path.join(self._cat_dir, line + ".tif")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.gimages.append(_g_image)
                self.categories.append(_cat)
                self.firename.append(line)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}
        if self.args.edge_detection:
            for split in self.split:
                if split == "train":
                    return self.merge2Tensor(self.transform_tr(sample), self.transform_tr_edge(sample))
                elif split == 'val':
                    return self.merge2Tensor(self.transform_val(sample), self.transform_val_edge(sample))
                else:
                    return self.merge2Tensor(self.transform_val(sample), self.transform_val_edge(sample))
        else:
            for split in self.split:
                if split == "train":
                    return self.transform_tr(sample)
                elif split == 'val':
                    return self.transform_val(sample)
                else:
                    return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _m_img = Image.open(self.gimages[index]).convert('RGB')

        _img = np.concatenate((_img, _m_img), axis=-1)

        _target = Image.open(self.categories[index])
        return _img, _target

    def merge2Tensor(self, sample1, sample2):
        img1 = sample1['image']
        mask1 = sample1['label']

        img2 = sample2['image']
        mask2 = sample2['label']

        img = torch.cat([img1, img2], dim=0)  # 变为6*340*360
        return {'image': img,
                'label': mask1}

    def transform_tr_edge(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            #             tr.RandomGaussianBlur(),
            #             tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.EdgeDetect(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val_edge(self, sample):

        composed_transforms = transforms.Compose([

            tr.FixScaleCrop(crop_size=self.args.crop_size),
            #             tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.EdgeDetect(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.ImsRandomHorizontalFlip(),
            tr.ImsRandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.ImsRandomGaussianBlur(),
            tr.ImsNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.ImsFixScaleCrop(crop_size=self.args.crop_size),
            tr.ImsNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'
