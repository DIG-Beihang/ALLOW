# Annealing-based-Label-Transfer-Learning-for-Open-World-Object-Detection

## Introduction

This repository is the official PyTorch implemetation of paper "**Annealing-based-Label-Transfer-Learning-for-Open-World-Object-Detection**".

![image](https://github.com/DIG-Beihang/Annealing-based-Label-Transfer-Learning-for-Open-World-Object-Detection/blob/master/docs/AnnealingOWOD.png)
<!---
<p align="center">
  <img src="./docs/framework.gif" alt="framework">
</p>
-->

**NOTE**: 
- In the code, We use the `cooling` variable to refer to the `extending` phase of a paper.
- In the `master` branch, we applied our method to the faster-rcnn framework, and in the `ow-detr` branch, we applied our method to the same deformable detr framework as ow-detr.
- If you want to learn more about the disentanglement and the visualization of our approach, please check out the [supplementary video](https://github.com/DIG-Beihang/Annealing-based-Label-Transfer-Learning-for-Open-World-Object-Detection/blob/master/docs/video%20(4).mp4).

## Key Code

Our code is based on the detectron2 framework to build, the main code directory is as follows： 

```
detectron2/
├── __init__.py
├── checkpoint
├── config
│   └── defaults.py
├── data
│   └── common.py
├── engine
│   └── defaults.py
├── evaluation
│   └── pascal_voc_evaluation.py
└── modeling
    ├── meta_arch
    │   └── rcnn.py
    └── roi_heads
        ├── fast_rcnn.py
        └── roi_heads.py
```
**Our method is simple to implement but very effective!!!**

**NOTE! You only need to enable our method during the extending stage.**

**If it is not the extending stage, simply set cfg.OWOD.COOLING = False to easily disable this feature.**
- First, modify the annotation of the data, we give all data an additional unknown class label, the code can be found [here](https://github.com/DIG-Beihang/Annealing-based-Label-Transfer-Learning-for-Open-World-Object-Detection/blob/ade50266435d699ece227192e08a46c26d57784f/detectron2/data/common.py#L52)
```
if self._map_func.is_train:
  data['instances'].ori_classes = data['instances'].gt_classes.clone()
  data['instances'].gt_classes[:] = 80
```
- Second, during the extending phase of the training, which is controlled by the cfg.OWOD.COOLING parameter, we need to freeze all parameters except for the classifier, and the specific code is [here](https://github.com/DIG-Beihang/ALL-OWOD/blob/c8bfcc8074407370184a48af58e20cdb22aa1aac/detectron2/engine/defaults.py#L285). Note that the optimizer in Detectron2 is initialized after this step, and during initialization, it ignores the parameters that do not have gradients.
```
if cfg.OWOD.COOLING:
  for name, param in model.named_parameters():
    if 'cls_score' not in name:
      param.requires_grad = False
```
- Finally, after calling the new loss function, you can start the training! Code can be found [here](https://github.com/DIG-Beihang/ALL-OWOD/blob/c8bfcc8074407370184a48af58e20cdb22aa1aac/detectron2/modeling/roi_heads/fast_rcnn.py#L334).
```
def mixup_loss(self):
  if self._no_instances:
    return 0.0 * self.pred_class_logits.sum()
  else:
    self._log_accuracy()
    self.pred_class_logits[:, self.invalid_class_range] = -10e10
    storage = get_event_storage()
    if self.cooling:
        lam = max(self.peak - (storage.iter / self.cool_iter), 0)
    else:
        lam = 0
    loss_pred = \
        (1-lam) * F.cross_entropy(self.pred_class_logits, self.ori_classes, reduction="mean", weight=self.weights) + \
        lam*F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean", weight=self.weights)
    return loss_pred
```

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


