"""
@author: Nguyen Duc "sh1nata" Tri <tri14102004@gmail.com>
"""

from torchvision.datasets import VOCSegmentation
import numpy as np
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VOCDataset(VOCSegmentation):
    # def __init__(self, root, year, image_set, download, transform = None, target_transform=None):
    #     super().__init__(root, year, image_set, download, transform, target_transform)
    #     self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    #                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    #                        'train', 'tvmonitor']

    def __init__(self, root, year, image_set, download, transform = None):
        super().__init__(root, year, image_set, download, transform)
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                           'train', 'tvmonitor']
        self.transform = transform 
    
    
    def __getitem__(self, item):
        image, target = super().__getitem__(item)
        target = np.array(target, np.int64)
        target[target == 255] = 0
       
        if self.transform:
        augmented = self.transform(image=image, mask=target)
        image = augmented['image']
        target = augmented['mask']
        
        return image, target

if __name__ == '__main__':
    # transform = Compose([
    #     Resize((224, 224)),
    #     ToTensor(),
    #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # target_transform = Resize((224, 224))
    # dateset = VOCDataset(root="../data", year="2012", image_set="train", download=False, transform=transform, target_transform=target_transform)

    transform = A.Compose([
        A.RandomScale(scale_limit=(0.5, 2.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    dateset = VOCDataset(root="../data", year="2012", image_set="train", download=False, transform=transform)

    image, target = dateset[1000]
    print(np.unique(target))
