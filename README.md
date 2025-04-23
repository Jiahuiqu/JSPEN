Deep Spatial-Spectral Joint-Sparse Prior Encoding Network for Hyperspectral Target Detection

The python code implementation of the paper "Deep Spatial-Spectral Joint-Sparse Prior Encoding Network for Hyperspectral Target Detection"
![image](https://github.com/Jiahuiqu/JSPEN/assets/78287811/4b1ad9cb-84a7-4f97-b5a2-32f42e8a3852)

### Requirements
    Ubuntu 18.04 cuda 11.0
    Python 3.7 Pytorch 1.11
### Usage
## Brief description
dataset floder stores training and testing dataset.
model floder stores the pretrained model for Texas Coast dataset.

## training
    run the train.py include train_background and train_target

## testing
    run the test.py to generate the loss map

### Citation
```
@ARTICLE{10549817,
  author={Dong, Wenqian and Wu, Xiaoyang and Qu, Jiahui and Gamba, Paolo and Xiao, Song and Vizziello, Anna and Li, Yunsong},
  journal={IEEE Transactions on Cybernetics}, 
  title={Deep Spatial—Spectral Joint-Sparse Prior Encoding Network for Hyperspectral Target Detection}, 
  year={2024},
  volume={54},
  number={12},
  pages={7780-7792},
  keywords={Object detection;Hyperspectral imaging;Adaptation models;Optimization;Training;Linear programming;Feature extraction;Hyperspectral target detection;interpretability;joint-sparse;model-encoding network},
  doi={10.1109/TCYB.2024.3403729}}
```
