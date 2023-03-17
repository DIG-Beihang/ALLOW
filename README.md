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

Our code is built based on the detectron2 framework, the main code directory is as follows, the left and right are the main code list of RCNN-based model and DETR-based model respectively： 

<html>
    <table style="width: 100%;">
        <tr>
            <td style="width: 50%;">
                <!--左侧内容-->
                <pre><code>
<strong>Faster rcnn-Based</strong> 
.
├── detectron2
│   ├── __init__.py
│   ├── checkpoint
│   ├── config
│   │   └── defaults.py
│   ├── data
│   │   └── LabelTrans_common.py
│   ├── engine
│   │   └── defaults.py
│   ├── evaluation
│   │   └── pascal_voc_evaluation.py
│   └── modeling
│       ├── meta_arch
│       │   └── rcnn.py
│       └── roi_heads
│          ├── AnneallingLT_out.py
│          └── AnneallingLT_heads.py
├── requirement.txt
├── setup.cfg
├── setup.py
└── scripts
    ├── test_all.sh
    ├── test_t1.sh
    ├── test_t2.sh
    ├── test_t3.sh
    ├── test_t4.sh
    ├── train_t1_extending.sh
    ├── train_t1.sh
    ├── train_t2_extending.sh
    ├── train_t2_ft.sh
    ├── train_t2.sh
    ├── train_t3_extending.sh
    ├── train_t3_ft.sh
    ├── train_t3.sh
    ├── train_t4_ft.sh
    └── train_t4.sh
            </code></pre>
            </td>
            <td style="width: 50%;">
                <!--右侧内容-->
                <pre><code>
<strong>DETR-Based</strong> 
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
│   └── AnneallingLT_detr.py
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
                </code></pre>
            </td>
        </tr>
    </table>
</html>


**Our method is simple to implement but very effective!!!**

**If it is not the extending stage, simply set cfg.OWOD.COOLING = False to easily disable this feature.**
- First, modify the annotation of the data, we give all data an additional unknown class label, the code can be found [here](https://github.com/DIG-Beihang/Annealing-based-Label-Transfer-Learning-for-Open-World-Object-Detection/blob/ade50266435d699ece227192e08a46c26d57784f/detectron2/data/LabelTrans_common.py#L52)
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
- Finally, after calling the new loss function, you can start the training! Code can be found [here](https://github.com/DIG-Beihang/ALL-OWOD/blob/c8bfcc8074407370184a48af58e20cdb22aa1aac/detectron2/modeling/roi_heads/AnneallingLT_out.py#L334).
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


