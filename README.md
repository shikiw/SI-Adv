# Shape-invariant 3D Adversarial Point Clouds (CVPR 2022)
> This repository provides the official PyTorch implementation of the following conference paper: <br>
> [**Shape-invariant 3D Adversarial Point Clouds (CVPR 2022)**](https://arxiv.org/abs/2203.04041) <br>
> **Qidong Huang, Xiaoyi Dong, Dongdong Chen, Weiming Zhang, Nenghai Yu.**
>



<p align="center"><img width="100%" src="teaser.png" /></p>

## Setup
The code is tested with Python3, Pytorch >= 1.6 and CUDA >= 10.2, including dependencies:

* tqdm >= 4.52.0
* numpy >= 1.19.2
* scipy >= 1.6.3
* open3d >= 0.13.0
* torchvision > =0.7.0
* scikit-learn >= 1.0

To complie cpp extension successfully, we list our dependencies for reference:

* gcc == 9.4.0
* ninja == 1.7.2

**TL;DR quick start**
We also provide a conda environment setup file including all of the above dependencies. Create the conda environment `si_adv_pc` by running:
```
conda env create -f environment.yml
```


## Preparation
Download the aligned [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) dataset and [ShapeNetPart](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip) dataset in their point cloud format and unzip them into your own dataset path.

Download the [pretrained models](https://drive.google.com/file/d/1L25i0l6L_b1Vw504WQR8-Z0oh2FJA0G9/view?usp=sharing) we provided for attack evaluation and unzip them at ```./checkpoint```.



## Citation
If you find this work useful for your research, please cite our [paper](https://arxiv.org/abs/2203.04041):
```
@article{huang2022siadv,
  title={Shape-invariant 3D Adversarial Point Clouds},
  author={Qidong Huang and Xiaoyi Dong and Dongdong Chen and Weiming Zhang and Nenghai Yu},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

