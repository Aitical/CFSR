<div align="center">

# [IEEE TIP] Transforming Image Super-Resolution: A ConvFormer-based Efficient Approach

[Gang Wu](https://scholar.google.com/citations?user=JSqb7QIAAAAJ), [Junjun Jiang](http://homepage.hit.edu.cn/jiangjunjun), [Junpeng Jiang](), and [Xianming Liu](http://homepage.hit.edu.cn/xmliu)

[AIIA Lab](https://aiialabhit.github.io/team/), Harbin Institute of Technology.

---

[![Paper2](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2401.05633)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20TIP-blue)](https://github.com/Aitical/CFSR) 
[![Models](https://img.shields.io/badge/Models-Hugging%20Face-gold)](https://huggingface.co/GWu/CFSR)
[![Results](https://img.shields.io/badge/Results-GoogleDrive-brightgreen)](https://drive.google.com/drive/folders/1M55TvlSn1BJVJ4Go5uVkvHFhfwo7Z5ov?usp=sharing)
[![Hits](https://hits.sh/github.com/Aitical/CFSR.svg)](https://hits.sh/github.com/Aitical/CFSR/)

</div>

## News
- Paper accepted by IEEE TIP. The final version is coming soon.
- [x] Upload implementation codes
- [x] Upload pretrained models


This repository is the official PyTorch implementation of "Transforming Image Super-Resolution: A ConvFormer-based Efficient Approach"

>Recent progress in single-image super-resolution (SISR) has achieved remarkable performance, yet the computational costs of these methods remain a challenge for deployment on resource-constrained devices. In particular, transformer-based methods, which leverage self-attention mechanisms, have led to significant breakthroughs but also introduce substantial computational costs. To tackle this issue, we introduce the Convolutional Transformer layer (ConvFormer) and propose a ConvFormer-based Super-Resolution network (CFSR), offering an effective and efficient solution for lightweight image super-resolution. The proposed method inherits the advantages of both convolution-based and transformer-based approaches. Specifically, CFSR utilizes large kernel convolutions as a feature mixer to replace the self-attention module, efficiently modeling long-range dependencies and extensive receptive fields with minimal computational overhead. Furthermore, we propose an edge-preserving feed-forward network (EFN) designed to achieve local feature aggregation while effectively preserving high-frequency information. Extensive experiments demonstrate that CFSR strikes an optimal balance between computational cost and performance compared to existing lightweight SR methods. When benchmarked against state-of-the-art methods such as ShuffleMixer, the proposed CFSR achieves a gain of 0.39 dB on the Urban100 dataset for the x2 super-resolution task while requiring 26\% and 31\% fewer parameters and FLOPs, respectively.


## Results

Results of x2, x3, and x4 SR tasks are available at [Google Drive](https://drive.google.com/drive/folders/1M55TvlSn1BJVJ4Go5uVkvHFhfwo7Z5ov?usp=sharing)
 
|Method | Scale| Params| FLOPs | Set5 (PSNR/SSIM)|Set14 (PSNR/SSIM)|B100 (PSNR/SSIM)|Urban100 (PSNR/SSIM)|Manga109 (PSNR/SSIM)|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VDSR  |  | $666 \mathrm{~K}$ | $612.6 \mathrm{G}$ | $31.35 / 0.8838$ | $28.01 / 0.7674$ | $27.29 / 0.7251$ | $25.18 / 0.7524$ | $28.83 / 0.8870$ |
| LapSRN  |  | $813 \mathrm{~K}$ | $149.4 \mathrm{G}$ | $31.54 / 0.8852$ | $28.09 / 0.7700$ | $27.32 / 0.7275$ | $25.21 / 0.7562$ | $29.09 / 0.8900$ |
| IDN  |  | $553 \mathrm{~K}$ | $32.3 \mathrm{G}$ | $31.82 / 0.8903$ | $28.25 / 0.7730$ | $27.41 / 0.7297$ | $25.41 / 0.7632$ | $29.41 / 0.8942$ |
| CARN  |  | $592 \mathrm{~K}$ | $90.9 \mathrm{G}$ | $32.13 / 0.8937$ | $28.60 / 0.7806$ | $27.58 / 0.7349$ | $26.07 / 0.7837$ | $30.47 / 0.9084$ |
| SRResNet|  | $1,518 \mathrm{~K}$ | $146 \mathrm{G}$ | $32.17 / 0.8951$ | $28.61 / 0.7823$ | $27.59 / 0.7365$ | $26.12 / 0.7871$ | $30.48 / 0.9087$ |
| IMDN |  | $715 \mathrm{~K}$ | $40.9 \mathrm{G}$ | $32.21 / 0.8948$ | $28.58 / 0.7811$ | $27.56 / 0.7353$ | $26.04 / 0.7838$ | $30.45 / 0.9075$ |
| LatticeNet |  | $777 \mathrm{~K}$ | $43.6 \mathrm{G}$ | $32.18 / 0.8943$ | $28.61 / 0.7812$ | $27.57 / 0.7355$ | $26.14 / 0.7844$ | $-/-$ |
| LAPAR-A  | $\times 4$ | $659 \mathrm{~K}$ | $94.0 \mathrm{G}$ | $32.15 / 0.8944$ | $28.61 / 0.7818$ | $27.61 / 0.7366$ | $26.14 / 0.7871$ | $30.42 / 0.9074$ |
| SMSR |  | $1006 \mathrm{~K}$ | $41.6 \mathrm{G}$ | $32.12 / 0.8932$ | $28.55 / 0.7808$ | $27.55 / 0.7351$ | $26.11 / 0.7868$ | $30.54 / 0.9085$ |
| ECBSR|  | $603 \mathrm{~K}$ | $34.7 \mathrm{G}$ | $31.92 / 0.8946$ | $28.34 / 0.7817$ | $27.48 / 0.7393$ | $25.81 / 0.7773$ | $-/-$ |
| PAN |  | $272 \mathrm{~K}$ | $28.2 \mathrm{G}$ | $32.13 / 0.8948$ | $28.61 / 0.7822$ | $27.59 / 0.7363$ | $26.11 / 0.7854$ | $30.51 / 0.9095$ |
| DRSAN |  | $410 \mathrm{~K}$ | $30.5 \mathrm{G}$ | $32.15 / 0.8935$ | $28.54 / 0.7813$ | $27.54 / 0.7364$ | $26.06 / 0.7858$ | $-/-$ |
| DDistill-SR  |  | $434 \mathrm{~K}$ | $33.0 \mathrm{G}$ | $32.23 / 0.8960$ | $28.62 / 0.7823$ | $27.58 / 0.7365$ | $26.20 / 0.7891$ | $30.48 / 0.9090$ |
| RFDN |  | $550 \mathrm{~K}$ | $23.9 \mathrm{G}$ | $32.24 / 0.8952$ | $28.61 / 0.7819$ | $27.57 / 0.7360$ | $26.11 / 0.7858$ | $30.58 / 0.9089$ |
| ShuffleMixer |  | $411 K$ | $28.0 \mathrm{G}$ | $32.21 / 0.8953$ | $28.66 / 0.7827$ | $27.61 / 0.7366$ | $26.08 / 0.7835$ | $30.65 / 0.9093$|
| **CFSR (Ours)** |  | $307 \mathrm{~K}$ | $17.5 \mathrm{G}$ |**$32.33/0.8964$**| **$28.73 / 0.7842$**| $27.63 / 0.7381$ | $26.21/0.7897$ | $30.72 / 0.9111$ |

## Citation
If you find this repository helpful, you may cite:
```
@ARTICLE{Wu_cfsr,
author={Wu, Gang and Jiang, Junjun and Junpeng Jiang and Liu, Xianming},
journal={IEEE Transactions on Image Processing}, 
title={Transforming Image Super-Resolution: A ConvFormer-based Efficient Approach}, 
year={2024},
}
```


## Acknowledgement

We thank the authors for their nice sharing of [BasicSR](https://github.com/XPixelGroup/BasicSR), [ECBSR](https://github.com/xindongzhang/ECBSR), and [ShuffleMixer](https://github.com/sunny2109/ShuffleMixer)




