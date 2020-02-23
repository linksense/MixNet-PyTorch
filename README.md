# MixNet-PyTorch
This repository contains a concise, modular, human-friendly **PyTorch** implementation of **[MixNet](https://arxiv.org/abs/1907.09595v3)** with **[Pre-trained Weights](https://github.com/ansleliu/MixNet-Pytorch/tree/master/pretrained_weights)**.


## Dependencies

- [PyTorch(1.4.1+)](http://pytorch.org)  
- [torchstat](https://github.com/Swall0w/torchstat)  
- [pytorch_memlab](https://github.com/Stonesjtu/pytorch_memlab)  


## Result Details(Val.)

|   *Name*   |*# Params*| *# FLOPS*  |*Top-1 Acc.*| *Pretrained* |
|:----------:|:--------:|:----------:|:----------:|:------------:|
| `MixNet-S` |   4.1M   |   0.256B   |    75.2    |   [GitHub](https://github.com/ansleliu/MixNet-Pytorch/blob/master/pretrained_weights/mixnet_s_top1v_75.2.pkl)   |
| `MixNet-M` |   5.0M   |   0.360B   |    76.5    |   [GitHub](https://github.com/ansleliu/MixNet-Pytorch/blob/master/pretrained_weights/mixnet_m_top1v_76.5.pkl)   |
| `MixNet-L` |   7.3M   |   0.565B   |    78.6    |   [GitHub](https://github.com/ansleliu/MixNet-Pytorch/blob/master/pretrained_weights/mixnet_l_top1v_78.6.pkl)   |
