# Annealing-based-Label-Transfer-Learning-for-Open-World-Object-Detection

## Introduction

This repository is the official PyTorch implemetation of paper "**Annealing-based-Label-Transfer-Learning-for-Open-World-Object-Detection**".

![image](https://github.com/DIG-Beihang/Annealing-based-Label-Transfer-Learning-for-Open-World-Object-Detection/blob/master/AnnealingOWOD.png)
; <p align="center">
;   <img src="./docs/framework.gif" alt="framework">
; </p>

**NOTE**: 
- In the code, We use the `cooling` variable to refer to the `extending` phase of a paper.
- In the `master` branch, we applied our method to the faster-rcnn framework, and in the `ow-detr` branch, we applied our method to the same deformable detr framework as ow-detr.
- If you want to learn more about the disentanglement and the visualization of our approach, please check out the [supplementary video]
(https://github.com/DIG-Beihang/Annealing-based-Label-Transfer-Learning-for-Open-World-Object-Detection/blob/master/video%20(4).mp4).

## Install
### Requirements
- Install detectron2, please refer to [INSTALL.md](./INSTALL.md).
- pip install -r requirements.txt
### Data Preparation for ORE split
- You can download the data set from [here](https://drive.google.com/drive/folders/1S5L-YmIiFMAKTs6nHMorB0Osz5iWI31k) and follow these steps to configure the path.
- Create folder datasets/VOC2007
- Put Annotations and JPEGImages inside datasets/VOC2007
- Create folder datasets/VOC2007/ImageSets/Main
- Put the content of datasets/OWOD_imagesets inside datasets/VOC2007/ImageSets/Main
### Pretrained weights
You can download the pre-trained backbone network model and the best OWOD models trained by ours methods for t1-t4 [here](https://drive.google.com/drive/folders/1baulMVqFWN-Vg_rVKJkkY3t_yAHtuhkJ?usp=sharing).
## Usage
### Training
- Download the pre-trained backbone network model. `R-50.pkl` is for faster rcnn framwork and `dino_resnet50_pretrain.pth` is for ow-detr framwork.
- Change the path to the pretrained model.
- You can sample run `train_*.sh` in `scripts` folder, Where `_t*_ `represents t1-t4 tasks, no endings represent the increment process of forming stage, and `ft` endings represent the fine-tuning process of forming stage, and `_extending` endings represent the extending stage.You should run in order such as:
```
bash train_t2.sh
bash train_t2_ft.sh
bash train_t2_extending.sh
```


### Evaluation
- You can sample run `test_*.sh` in `scripts` folder.

## Citation

If this work helps your research, please cite the following paper.


