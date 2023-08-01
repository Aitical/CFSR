<div align="center">

# CFSR: A Simple Convolutional Transformer for Lightweight Image Super-Resolution

[Gang Wu](https://scholar.google.com/citations?user=JSqb7QIAAAAJ), [Junjun Jiang](http://homepage.hit.edu.cn/jiangjunjun), [Junpeng Jiang](), and [Xianming Liu](http://homepage.hit.edu.cn/xmliu)

[AIIA Lab](https://aiialabhit.github.io/team/), Harbin Institute of Technology.

---
[paper]()
**|**
[pretrained models]()

[![Hits](https://hits.sh/github.com/Aitical/CFSR.svg)](https://hits.sh/github.com/Aitical/CFSR/)

</div>

This repository is the official PyTorch implementation of "CFSR: A Simple Convolutional Transformer for Lightweight Image Super-Resolution"

>Recent progress in single-image super-resolution (SISR) has achieved remarkable performance, yet computational costs remain a challenge for deployment on resource-constrained devices. Especially for transformer-based methods, the self-attention mechanism in such models brings great breakthroughs while incurring substantial computational costs. To address this problem, we introduce the Convolutional Transformer layer (ConvFormer) and the ConvFormer-based Super-Resolution network (CFSR). Our CFSR architecture presents an effective and efficient solution for lightweight image super-resolution tasks. In detail, CFSR leverages the large kernel convolution as the feature mixer to replace the self-attention module, efficiently modeling long-range dependencies and extensive receptive fields with a slight computational cost. Furthermore, we propose an edge-preserving feed-forward network, simplified as EFN, to obtain local feature aggregation and simultaneously preserve more high-frequency information. Extensive experiments demonstrate that CFSR can achieve an advanced trade-off between computational cost and performance when compared to existing lightweight SR methods. Compared to state-of-the-art methods, e.g. ShuffleMixer, the proposed CFSR achieves **0.39** dB gains on Urban100 dataset for x2 SR task while containing **26%** and **31%** fewer parameters and FLOPs, respectively. 

## TODO
- [ ] Add implementation code
- [ ] Add pretrained model
- [ ] Add results of test set

