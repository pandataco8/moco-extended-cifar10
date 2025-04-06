import torch
import random
from PIL import ImageFilter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class TwoCropsTransform:
   def __init__(self, base_transform):
       self.base_transform = base_transform


   def __call__(self, x):
       return [self.base_transform(x), self.base_transform(x)]


class GaussianBlur:
   def __init__(self, sigma=[0.1, 2.0]):
       self.sigma = sigma


   def __call__(self, x):
       sigma = random.uniform(self.sigma[0], self.sigma[1])
       return x.filter(ImageFilter.GaussianBlur(radius=sigma))

class RandomErasingTransform(object):
   def __init__(self, p=0.25):
       self.erase = transforms.RandomErasing(p=p, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')


   def __call__(self, img_tensor):
       return self.erase(img_tensor)


def get_augmented_loader(batch_size, augmentation_type='default', workers=4):

   train_dataset = datasets.CIFAR10(
       root='./data',
       train=True,
       download=True,
       transform=TwoCropsTransform(get_transform(augmentation_type))
   )


   return torch.utils.data.DataLoader(
       train_dataset, batch_size=batch_size, shuffle=True,
       num_workers=workers, pin_memory=True, drop_last=True)


def get_transform(augmentation_type='default'):

   if augmentation_type == 'aggressive':
       print("=> Using aggressive augmentations")
       return transforms.Compose([
           transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
           transforms.RandomApply([
               transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
           ], p=0.8),
           transforms.RandomGrayscale(p=0.2),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           RandomErasingTransform(p=0.25),
       ])
   else:
       print("=> Using default augmentations")
       return transforms.Compose([
           transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
           transforms.RandomGrayscale(p=0.2),
           transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
       ])



