# Deeplabv3-Pytorch
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
After training 80 epochs: <br>
+ Model: deeplabv3_mobilenet_v3_large (could improve further by training 100 epochs) <br>
 
![image](https://github.com/user-attachments/assets/7db2b280-9215-4fad-8486-730f45b631d4)


