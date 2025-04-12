# Deeplabv3-Pytorch
Semantic Image Segmentation using Deeplabv3

## Introduction 
Here is my pytorch implementation of the model described in the paper **Rethinking Atrous Convolution for Semantic Image Segmentation** [paper](https://arxiv.org/pdf/1706.05587v3). 

# Requirements: 

# Datasets: 
I used 2 different datasets: PascalVOC2012 and COCO 

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
