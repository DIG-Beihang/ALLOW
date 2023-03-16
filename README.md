# Annealing-based-Label-Transfer-Learning-for-Open-World-Object-Detection

## Introduction

This repository is the official PyTorch implemetation of paper "**Annealing-based-Label-Transfer-Learning-for-Open-World-Object-Detection**".

![image](https://github.com/DIG-Beihang/Annealing-based-Label-Transfer-Learning-for-Open-World-Object-Detection/blob/master/docs/AnnealingOWOD.png)
<p align="center">
  <img src="https://github.com/DIG-Beihang/Annealing-based-Label-Transfer-Learning-for-Open-World-Object-Detection/blob/master/docs/framework.gif" alt="framework">
</p>

**NOTE**: 
- In the code, We use the `cooling` variable to refer to the `extending` phase of a paper.
- In the `master` branch, we applied our method to the faster-rcnn framework, and in the `ow-detr` branch, we applied our method to the same deformable detr framework as ow-detr

## Key Code
```
.
├── configs
│   └── new1026
│       ├── OWOD_new_split_eval.sh
│       ├── OWOD_new_split_eval_t1_NC.sh
│       ├── OWOD_new_split_eval_t2.sh
│       ├── OWOD_new_split_eval_t3.sh
│       ├── OWOD_ore_split_t1.sh
│       ├── OWOD_ore_split_t1_extending.sh
│       ├── OWOD_ore_split_t2.sh
│       ├── OWOD_ore_split_t2_extending.sh
│       ├── OWOD_ore_split_t2ft.sh
│       ├── OWOD_ore_split_t3.sh
│       ├── OWOD_ore_split_t3_extending.sh
│       ├── OWOD_ore_split_t3ft.sh
│       ├── OWOD_ore_split_t4.sh
│       └── OWOD_ore_split_t4ft.sh
├── main_open_world.py
├── models
│   └── deformable_detr.py
├── requirements.txt
└── scripts
    ├── test_t1.sh
    ├── test_t1_NC.sh
    ├── test_t2.sh
    ├── test_t2ft.sh
    ├── test_t3.sh
    ├── train_t1.sh
    ├── train_t1_extending.sh
    ├── train_t2.sh
    ├── train_t2_extending.sh
    ├── train_t2_ft.sh
    ├── train_t3.sh
    ├── train_t3_extending.sh
    ├── train_t3_ft.sh
    ├── train_t4.sh
    └── train_t4_ft.sh
```
- We did not use the relevant innovations of ow-detr. Specifically, we removed --unmatched_boxes, --NC_branch, --nc_loss_coef, --top_unk, and --nc_epoch from the script to train the closed-set model, and added --cooling and --cooling_prev to train the extending stage model.
- During the extending phase of the training, which is controlled by the args.cooling parameter, we need to freeze all parameters except for the classifier, and the specific code is [here](https://github.com/DIG-Beihang/ALL-OWOD/blob/5f05d39f9c6f6edc405eb269be720d5a291b2424/main_open_world.py#L163). Note that the optimizer is initialized after this step, and during initialization, it ignores the parameters that do not have gradients.
```
if args.cooling:
  print('-------------------------------cooling------------------------------------')
  for name, param in model_without_ddp.named_parameters():
    if not 'class_embed' in name:
      param.requires_grad = False
```
- Add the following code to loss_labels, you can start the training! Code can be found [here](https://github.com/DIG-Beihang/ALL-OWOD/blob/5f05d39f9c6f6edc405eb269be720d5a291b2424/models/deformable_detr.py#L350)
```
target_classesAL = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
target_classes_u = torch.full(target_classes_o.size(), self.num_classes - 1, dtype=torch.int64, device=src_logits.device)
target_classesAL[idx] = target_classes_u
target_classes_onehotAL = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                    dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
target_classes_onehotAL.scatter_(2, target_classesAL.unsqueeze(-1), 1)
target_classes_onehotAL = target_classes_onehotAL[:,:,:-1]
loss_ce_u = sigmoid_focal_loss(src_logits, target_classes_onehotAL, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

lam = 0
if self.cooling:
    lam = max(1 - (current_epoch - self.cooling_prev) / 20, 0)
loss_ce = (1-lam) * loss_ce + lam * loss_ce_u
```

## Install
### Requirements
We have trained and tested our models on `Ubuntu 16.0`, `CUDA 10.2`, `GCC 5.4`, `Python 3.7`

```bash
conda create -n owdetr python=3.7 pip
conda activate owdetr
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```
### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```
### Data Preparation for ORE split
- You can download the data set from [here](https://drive.google.com/drive/folders/1S5L-YmIiFMAKTs6nHMorB0Osz5iWI31k) and follow these steps to configure the path.
The files should be organized in the following structure:
```
OW-DETR/
└── data/
    └── VOC2007/
        └── OWOD/
        	├── JPEGImages
        	├── ImageSets
        	└── Annotations
```
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


