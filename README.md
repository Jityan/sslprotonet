# SSL-ProtoNet: Self-supervised Learning Prototypical Networks for Few-shot Learning

This repository contains the **pytorch** code for the under review paper: "SSL-ProtoNet: Self-supervised Learning Prototypical Networks for Few-shot Learning" Jit Yan Lim, Kian Ming Lim, Chin Poo Lee, Yong Xuan Tan

## Environment
The code is tested on Windows 10 with Anaconda3 and following packages:
- python 3.7.4
- pytorch 1.3.1

## Preparation
1. Change the ROOT_PATH value in the following files to yours:
    - `datasets/mini_imagenet.py`
    - `datasets/tiered_imagenet.py`
    - `datasets/cifarfs.py`

2. Download the datasets and put them into corresponding folders that mentioned in the ROOT_PATH:<br/>
    - ***mini*ImageNet**: download from [CSS](https://github.com/anyuexuan/CSS) and put in `data/mini-imagenet` folder.

    - ***tiered*ImageNet**: download from [MetaOptNet](https://github.com/kjunelee/MetaOptNet) and put in `data/tiered-imagenet` folder.

    - **CIFARFS**: download from [MetaOptNet](https://github.com/kjunelee/MetaOptNet) and put in `data/cifar-fs` folder.


## Experiments
To train on 1-shot and 5-shot CIFAR-FS:<br/>
```
python train_stage1.py --dataset cifarfs --train-way 50 --train-batch 100 --save-path ./save/cifarfs-stage1

python train_stage2.py --dataset cifarfs --shot 1 --save-path ./save/cifarfs-stage2-1s --stage1-path ./save/cifarfs-stage1 --train-way 20
python train_stage2.py --dataset cifarfs --shot 5 --save-path ./save/cifarfs-stage2-5s --stage1-path ./save/cifarfs-stage1 --train-way 10

python train_stage3.py --kd-coef 0.7 --dataset cifarfs --shot 1 --train-way 20 --stage1-path ./save/cifarfs-stage1 --stage2-path ./save/cifarfs-stage2-1s --save-path ./save/cifarfs-stage3-1s
python train_stage3.py --kd-coef 0.1 --dataset cifarfs --shot 5 --train-way 10 --stage1-path ./save/cifarfs-stage1 --stage2-path ./save/cifarfs-stage2-5s --save-path ./save/cifarfs-stage3-5s
```
To evaluate on 5-way 1-shot and 5-way 5-shot CIFAR-FS:<br/>
```
python test.py --dataset cifarfs --shot 1 --save-path ./save/cifarfs-stage3-1s
python test.py --dataset cifarfs --shot 5 --save-path ./save/cifarfs-stage3-1s
```

## Contacts
For any questions, please contact: <br/>

Jit Yan Lim (lim.jityan@mmu.edu.my) <br/>
Kian Ming Lim (kmlim@mmu.edu.my)

## Acknowlegements
This repo is based on **[Prototypical Networks](https://github.com/yinboc/prototypical-network-pytorch)**, **[RFS](https://github.com/WangYueFt/rfs)**, and **[SKD](https://github.com/brjathu/SKD)**.
