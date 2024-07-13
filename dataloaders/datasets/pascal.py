from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
import torch

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
        base_dir = Path.db_root_dir(self.args.dataset)
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split


        self.NUM_CLASSES = self.args.num_class

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.firename = []
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):

                _image = os.path.join(self._image_dir, line + ".tif")
                # print(_image)
                if not os.path.isfile(_image):
                    _image = os.path.join(self._image_dir, line + "_" + self.args.suffix[0] + ".tif")
                    # print(_image)
                _cat = os.path.join(self._cat_dir, line + ".tif")
                # print(_cat)
                # print(_cat)
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
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
        _target = Image.open(self.categories[index])
        return _img, _target

    def merge2Tensor(self, sample1, sample2):
        img1 = sample1['image']
        mask1 = sample1['label']
        
        img2 = sample2['image']
        mask2 = sample2['label']
        
        img = torch.cat([img1, img2], dim=0) #变为6*340*360
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
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


class CustomVOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
  

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('pascalpred'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
          #  这里我们的数据集应该是15分类，加上背景总共16分类

        super().__init__()
        self._base_dir = base_dir
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
        self.categories = []
        self.firename = []
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".png")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
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
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr_edge(self, sample):
        composed_transforms = transforms.Compose([
            tr.EdgeDetect(),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
#             tr.RandomGaussianBlur(),
#             tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
            
        return composed_transforms(sample)
    
    def transform_val_edge(self, sample):

        composed_transforms = transforms.Compose([
            
            tr.FixScaleCrop(crop_size=self.args.crop_size),
#             tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.EdgeDetect(),
            tr.ToTensor()])
        
        return composed_transforms(sample)

    def merge2Tensor(self, sample1, sample2):
        img1 = sample1['image']
        mask1 = sample1['label']
        
        img2 = sample2['image']
        mask2 = sample2['label']
        
        img = torch.cat([img1, img2], dim=0) #变为6*340*360
        return {'image': img,
                'label': mask1}
    
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'
if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


