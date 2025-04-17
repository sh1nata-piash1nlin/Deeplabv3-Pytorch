Semantic Image Segmentation using Deeplabv3

# Introduction 
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

# Training Protocol: 
* **Optimizer and Learning Rate:**
  + Adam Optimizer with learning rate 0.001. 
  + SGD Optimizer with different learning rate at different epochs (learning rate 0.01 in most cases).
If you want to train with different settings above, for example you can: 
```sh
    python trainVOCDataset.py -op sgd 
```
* **Data Augmentation:**
  + The default requirments of pytorch (Resize, ToTensor)
  + Add-on using `Albumentation`: RandomScale, HorizontalFlip, ToTensorV2() which are proposed in the paper.
  
# Result of training: 
+ Model: deeplabv3_mobilenet_v3_large (epoch = 130, batch_size=2, image_size = (224, 224)): <br> 

![image](https://github.com/user-attachments/assets/ffb2d456-7e28-43b7-866f-2d0c06e938ae)

+ Model: deeplabv3_resnet101 (epoch = , batch_size=2, image_size = (513, 513)): <br>

