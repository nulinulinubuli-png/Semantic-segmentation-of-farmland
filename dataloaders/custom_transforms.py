import torch
import random
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter

class MsNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        _img = img[:,:,0:3]
        m_img = img[:,:,3:6]
        l_img = img[:,:,6:9]
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        _img -= self.mean
        l_img -= self.mean
        m_img -= self.mean
        
        _img /= self.std
        l_img /= self.std
        m_img /= self.std
        
        img = np.concatenate((_img, m_img, l_img), axis=2)

        return {'image': img,
                'label': mask}


class ImsNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        _img = img[:, :, 0:3]
        m_img = img[:, :, 3:6]
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        _img -= self.mean
        m_img -= self.mean

        _img /= self.std
        m_img /= self.std

        img = np.concatenate((_img, m_img), axis=2)

        return {'image': img,
                'label': mask}

class ImsNormalizev4(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        _img = img[:, :, 0:3]
        m_img = img[:, :, 3:6]
        image_hom = img[:, :, 6:9]
        image_orgin = img[:, :, 9:12]
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        _img -= self.mean
        m_img -= self.mean

        _img /= self.std
        m_img /= self.std

        image_hom -= self.mean
        image_orgin -= self.mean


        image_hom /= self.std
        image_orgin /= self.std

        img = np.concatenate((_img, m_img,image_hom,image_orgin), axis=2)

        return {'image': img,
                'label': mask}

#三张纹理特征图
class ImsNormalize3(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        image_entropy = img[:, :, 0:3]
        image_diss = img[:, :, 3:6]
        image_hom = img[:, :, 6:9]
        image_orgin = img[:, :, 9:12]
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        image_entropy -= self.mean
        image_diss -= self.mean
        image_hom -= self.mean
        image_orgin -= self.mean

        image_entropy /= self.std
        image_diss /= self.std
        image_hom /= self.std
        image_orgin /= self.std

        img = np.concatenate((image_entropy,image_diss,image_hom,image_orgin), axis=2)

        return {'image': img,
                'label': mask}

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}

class EdgeDetect(object):
    """边缘检测"""
    def __call__(self, sample):
        # numpy image: H x W x C
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.uint8).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        #分别对RGB三通道进行边缘检测
        res = []
        for channel in img:
            tmp1 = cv2.GaussianBlur(channel, (3,3), 0)
            canny = cv2.Canny(tmp1, 180, 220)
            res.append(canny)
        img = np.array(res).astype(np.float32)
        img = img.transpose((1,2,0))
        img /= 255.0
        return {'image': img,
                'label': mask}



    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}

#图像左右翻转
class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)  
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class MsRandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = np.fliplr(img)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class ImsRandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = np.fliplr(img)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}
#图像旋转
class MsRandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        _img = Image.fromarray(img[:,:,0:3]).rotate(rotate_degree, Image.BILINEAR)
        m_img = Image.fromarray(img[:,:,3:6]).rotate(rotate_degree, Image.BILINEAR)
        l_img = Image.fromarray(img[:,:,6:9]).rotate(rotate_degree, Image.BILINEAR)
        img = np.concatenate((_img, m_img, l_img), axis=2)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}
#图像旋转
class ImsRandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        _img = Image.fromarray(img[:,:,0:3]).rotate(rotate_degree, Image.BILINEAR)
        m_img = Image.fromarray(img[:,:,3:6]).rotate(rotate_degree, Image.BILINEAR)
        img = np.concatenate((_img, m_img), axis=2)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}
    
class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}
#图像高斯滤波
class MsRandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            _img = Image.fromarray(img[:,:,0:3]).filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            m_img = Image.fromarray(img[:,:,3:6]).filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            l_img = Image.fromarray(img[:,:,6:9]).filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            img = np.concatenate((_img, m_img, l_img), axis=2)

        return {'image': img,
                'label': mask}
#图像高斯滤波
class ImsRandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            _img = Image.fromarray(img[:,:,0:3]).filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            m_img = Image.fromarray(img[:,:,3:6]).filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            img = np.concatenate((_img, m_img), axis=2)

        return {'image': img,
                'label': mask}

class ImsRandomGaussianBlurv4(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            _img = Image.fromarray(img[:,:,0:3]).filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            m_img = Image.fromarray(img[:,:,3:6]).filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            image_hom = Image.fromarray(img[:, :, 6:9]).filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            image_orgin = Image.fromarray(img[:, :, 9:12]).filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            img = np.concatenate((_img, m_img, image_hom, image_orgin), axis=2)

        return {'image': img,
                'label': mask}
#三张纹理特征图像高斯滤波
class ImsRandomGaussianBlur3(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            image_entropy = Image.fromarray(img[:,:,0:3]).filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            image_diss = Image.fromarray(img[:,:,3:6]).filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            image_hom = Image.fromarray(img[:, :, 6:9]).filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            image_orgin = Image.fromarray(img[:, :, 9:12]).filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            img = np.concatenate((image_entropy, image_diss, image_hom, image_orgin), axis=2)

        return {'image': img,
                'label': mask}

class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}
#随机尺寸裁剪
class MsRandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
#         print(img.size)
        w, h,_ = img.shape
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        _img = Image.fromarray(img[:,:,0:3])
        m_img = Image.fromarray(img[:,:,3:6])
        l_img = Image.fromarray(img[:,:,6:9])
        _img = _img.resize((ow, oh), Image.BILINEAR)
        m_img = m_img.resize((ow, oh), Image.BILINEAR)
        l_img = l_img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            _img = ImageOps.expand(_img, border=(0, 0, padw, padh), fill=0)
            m_img = ImageOps.expand(m_img, border=(0, 0, padw, padh), fill=0)
            l_img = ImageOps.expand(l_img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = _img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        _img = _img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        m_img = m_img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        l_img = l_img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        img = np.concatenate((_img, m_img, l_img), axis=2)
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class ImsRandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        #         print(img.size)
        w, h, _ = img.shape
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        _img = Image.fromarray(img[:, :, 0:3])
        m_img = Image.fromarray(img[:, :, 3:6])
        _img = _img.resize((ow, oh), Image.BILINEAR)
        m_img = m_img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            _img = ImageOps.expand(_img, border=(0, 0, padw, padh), fill=0)
            m_img = ImageOps.expand(m_img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = _img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        _img = _img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        m_img = m_img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        img = np.concatenate((_img, m_img), axis=2)
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}
class ImsRandomScaleCropv4(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        #         print(img.size)
        w, h, _ = img.shape
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        _img = Image.fromarray(img[:, :, 0:3])
        m_img = Image.fromarray(img[:, :, 3:6])
        image_hom = Image.fromarray(img[:, :, 6:9])
        image_orgin = Image.fromarray(img[:, :, 9:12])
        _img = _img.resize((ow, oh), Image.BILINEAR)
        m_img = m_img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            _img = ImageOps.expand(_img, border=(0, 0, padw, padh), fill=0)
            m_img = ImageOps.expand(m_img, border=(0, 0, padw, padh), fill=0)
            image_hom = ImageOps.expand(image_hom, border=(0, 0, padw, padh), fill=0)
            image_orgin = ImageOps.expand(image_orgin, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = _img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        _img = _img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        m_img = m_img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        image_hom = image_hom.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        image_orgin = image_orgin.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        img = np.concatenate((_img, m_img,image_hom,image_orgin), axis=2)
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

# 三张纹理特征图的拼接
class ImsRandomScaleCrop3(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        #         print(img.size)
        w, h, _ = img.shape
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        image_entropy = Image.fromarray(img[:, :, 0:3])
        image_diss = Image.fromarray(img[:, :, 3:6])
        image_hom = Image.fromarray(img[:, :, 6:9])
        image_orgin = Image.fromarray(img[:, :, 9:12])
        image_entropy = image_entropy.resize((ow, oh), Image.BILINEAR)
        image_diss = image_diss.resize((ow, oh), Image.BILINEAR)
        image_hom = image_hom.resize((ow, oh), Image.BILINEAR)
        image_orgin = image_orgin.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            image_entropy = ImageOps.expand(image_entropy, border=(0, 0, padw, padh), fill=0)
            image_diss = ImageOps.expand(image_diss, border=(0, 0, padw, padh), fill=0)
            image_hom = ImageOps.expand(image_hom, border=(0, 0, padw, padh), fill=0)
            image_orgin = ImageOps.expand(image_orgin, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = image_orgin.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        image_entropy = image_entropy.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        image_diss = image_diss.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        image_hom = image_hom.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        image_orgin = image_orgin.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        img = np.concatenate((image_entropy, image_diss, image_hom, image_orgin), axis=2)
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # print("开始进行三张特征图拼接")

        return {'image': img,
                'label': mask}

#随机尺寸裁剪
class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
#         print(img.size)
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

    #修正尺寸大小裁剪
class MsFixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        _img = Image.fromarray(img[:,:,0:3])
        m_img = Image.fromarray(img[:,:,3:6])
        l_img = Image.fromarray(img[:,:,6:9])
        w, h,_ = img.shape
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        _img = _img.resize((ow, oh), Image.BILINEAR)
        m_img = m_img.resize((ow, oh), Image.BILINEAR)
        l_img = l_img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = _img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        _img = _img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        m_img = m_img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        l_img = l_img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img = np.concatenate((_img, m_img, l_img), axis=2)
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

 #修正尺寸大小裁剪
class ImsFixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        _img = Image.fromarray(img[:,:,0:3])
        m_img = Image.fromarray(img[:,:,3:6])
        w, h,_ = img.shape
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        _img = _img.resize((ow, oh), Image.BILINEAR)
        m_img = m_img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = _img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        _img = _img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        m_img = m_img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img = np.concatenate((_img, m_img), axis=2)
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return {'image': img,
                'label': mask}

class ImsFixScaleCropv4(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        _img = Image.fromarray(img[:,:,0:3])
        m_img = Image.fromarray(img[:,:,3:6])
        image_hom = Image.fromarray(img[:, :, 6:9])
        image_orgin = Image.fromarray(img[:, :, 9:12])
        w, h,_ = img.shape
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        _img = _img.resize((ow, oh), Image.BILINEAR)
        m_img = m_img.resize((ow, oh), Image.BILINEAR)
        image_hom = image_hom.resize((ow, oh), Image.BILINEAR)
        image_orgin = image_orgin.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = _img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        _img = _img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        m_img = m_img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        image_hom = image_hom.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        image_orgin = image_orgin.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img = np.concatenate((_img, m_img,image_hom,image_orgin), axis=2)
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return {'image': img,
                'label': mask}


# 三张纹理特征图的修正尺寸大小裁剪
class ImsFixScaleCrop3(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        image_entropy = Image.fromarray(img[:, :, 0:3])
        image_diss = Image.fromarray(img[:, :, 3:6])
        image_hom = Image.fromarray(img[:, :, 6:9])
        image_orgin = Image.fromarray(img[:, :, 9:12])
        w, h,_ = img.shape
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        image_entropy = image_entropy.resize((ow, oh), Image.BILINEAR)
        image_diss = image_diss.resize((ow, oh), Image.BILINEAR)
        image_hom = image_hom.resize((ow, oh), Image.BILINEAR)
        image_orgin = image_orgin.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = image_orgin.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        image_entropy = image_entropy.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        image_diss = image_diss.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        image_hom = image_hom.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        image_orgin = image_orgin.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img = np.concatenate((image_entropy, image_diss, image_hom, image_orgin), axis=2)
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return {'image': img,
                'label': mask}

#修正尺寸大小裁剪
class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}
#图像resize
class MsFixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        image_entropy = Image.fromarray(img[:,:,0:3])
        image_diss = Image.fromarray(img[:,:,3:6])
        image_hom = Image.fromarray(img[:,:,6:9])
        image_orgin = Image.fromarray(img[:,:,9:12])
        assert image_entropy.size == mask.size
        assert image_orgin.size == mask.size
        
        image_entropy = image_entropy.resize(self.size, Image.BILINEAR)
        image_diss = image_diss.resize(self.size, Image.BILINEAR)
        image_hom = image_hom.resize(self.size, Image.BILINEAR)
        image_orgin = image_orgin.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        
        img = np.concatenate((image_entropy, image_diss, image_hom, image_orgin), axis=2)
        return {'image': img,
                'label': mask}
#图像resize
class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}