![image](https://github.com/user-attachments/assets/3a5a08ab-d0a0-44c4-8c65-496a16217ad3)# Deeplabv3-Pytorch
Semantic Image Segmentation using Deeplabv3

## Introduction 
Here is my pytorch implementation of the model described in the paper **Rethinking Atrous Convolution for Semantic Image Segmentation** [paper](https://arxiv.org/pdf/1706.05587v3). 

# Requirements: 
+ python 3.6
+ pytorch 0.4
+ openCV
+ torchvision
+ PIL
+ numpy
+ tensorboard

# Datasets: 
I used 2 different datasets: PascalVOC2012 and COCO. <br> 
In my **src** folder, going to the code which relates to the dataset you want to download, changing the parameter **download** into **True** to download the dataset you need. <br>

```sh
data
├── VOC2012
│   ├── Annotations  
│   ├── ImageSets
│   ├── JPEGImages
│   └── ...
```

# Training: 
If you want to train a new model, you could run:
```sh
    python trainVOCDataset.py --dataset dataset: For example, python train_voc.py --dataset VOC2012
```

# Result of traning: 
+ Model: deeplabv3_mobilenet_v3_large (epoch = 130, batch_size=2, image_size = (224, 224): <br> 

![image](https://github.com/user-attachments/assets/33afd0bd-236f-4ec7-aa85-24e67159bcd4)

+ Model: deeplabv3_resnet101 (epoch = , batch_size=2, image_size = (513, 513)): <br>

