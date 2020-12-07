## Heterogeneous Double-Head Ensemble for Deep Metric Learning

Official Pytorch implementation of paper:

[Heterogeneous Double-Head Ensemble for Deep Metric Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9123761) (IEEE Access 2020).



## Environment
Python 3.6, Pytorch 0.4.1, Torchvision, tensorboard


## Train 
Default setting:
- Architecture: ResNet-50
- Dataset: CUB2011 (or Cars-196, Inshop, SOP)
- Batch size: 40
- Image size: 224X224


### prepare
The dataset path should be changed to your own path.

CUB2011-200 dataset are available on https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view

Cars-196 dataset are available on https://ai.stanford.edu/~jkrause/cars/car_dataset.html

```
prepare_cub.py 
```

### train network. 

The dataset path(data_dir='/home/ro/FG/CUB_200_2011/pytorch') should be changed to your own path.


```
HDhE_train.py --dataset CUB-200
```

In the case of Cars-196 retrieval dataset training, 

```
HDhE_train.py --dataset Cars-196
```





## Citation

```
@ARTICLE{9123761,
  author={Y. {Ro} and J. Y. {Choi}},
  journal={IEEE Access}, 
  title={Heterogeneous Double-Head Ensemble for Deep Metric Learning}, 
  year={2020},
  volume={8},
  number={},
  pages={118525-118533},
  doi={10.1109/ACCESS.2020.3004579}}
```
Youngmin Ro, Jin Young Choi, 
"Heterogeneous Double-Head Ensemble for Deep Metric Learning", IEEE Access, 2020.



