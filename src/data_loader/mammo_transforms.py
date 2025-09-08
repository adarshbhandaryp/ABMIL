from random import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import random
import math
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

def random_shear(shear_range):
    shear = random.uniform(-shear_range, shear_range)
    shear_matrix = np.array([[1, -math.sin(shear), 0],
                             [0, math.cos(shear), 0],
                             [0, 0, 1]])
    return shear_range


def random_zoom(zoom_range):
    zx = random.uniform(zoom_range[0], zoom_range[1])
    zy = random.uniform(zoom_range[0], zoom_range[1])
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    return zoom_matrix



class TrainTransformBaseline:
    def __init__(self, aug: bool, aug_mix_p : int, erasing : int):
        self.aug = aug
        self.aug_mix_p = aug_mix_p
        self.erasing = erasing

        if self.aug:
            self.data_transforms_train = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomEqualize(p=0.1),
                transforms.RandomApply(torch.nn.ModuleList([
                       transforms.AugMix(),
                   ]), p=self.aug_mix_p),
                transforms.ToTensor(),
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                #transforms.RandomErasing(p=self.erasing, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            ])

        else:
            self.data_transforms_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])

    def __call__(self, x):
        y = self.data_transforms_train(x)
        return y


class TrainTransform:
    def __init__(self, aug: bool, aug_mix_p : int):
        self.aug = aug
        self.aug_mix_p = aug_mix_p

        if self.aug:
            self.data_transforms_train = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomEqualize(p=0.2),
                transforms.RandomApply(torch.nn.ModuleList([
                       transforms.AugMix(),
                   ]), p=self.aug_mix_p),
                transforms.ToTensor(),
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                #transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            ])

        else:
            self.data_transforms_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])

    def __call__(self, x):
        y = self.data_transforms_train(x)
        return y


class ValidTransform:
    def __init__(self, input_size):
        self.input_size = input_size
        self.data_transforms_validation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor()
        ])

    def __call__(self, x):
        y = self.data_transforms_validation(x)
        return y


class TestTransform:
    def __init__(self, input_size, probability):
        self.input_size = input_size
        self.probability = probability
        self.data_transforms_validation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])
        self.data_transforms_validation_p1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            # transforms.RandomEqualize(p=1.0),
            transforms.ToTensor()
        ])

    def __call__(self, x):
        if self.probability == 1:
            y = self.data_transforms_validation_p1(x)
        else:
            y = self.data_transforms_validation(x)
        return y


class TrainTransformSupCon:
    def __init__(self, input_size):
        self.input_size = input_size
        self.data_transforms_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.RandomResizedCrop(size=self.input_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomEqualize(p=0.4),
            # transforms.RandomApply(torch.nn.ModuleList([
            #        transforms.RandomRotation(180),
            #    ]), p=0.5),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()
        ])

    def __call__(self, x):
        y = self.data_transforms_train(x)
        return y


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class TestTimeAugmentationTransforms:

    def __init__(self, input_size):
        self.input_size = input_size
        self.data_transforms_validation_tta = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ])

    def __call__(self, x):
        y = self.data_transforms_validation_tta(x)
        return y

class BasicTransforms:

    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize((224,224)),
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __call__(self, x):
        y = self.train_transform(x)
        return y