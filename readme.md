# Keras implementation of EEDS: End-to-End image super-resolution via deep and shallow convolutional networks


The original paper is [end-to-end image super-resolution via deep and shallow convolutional networks](https://arxiv.org/abs/1607.07680)

<p align="center">
  <img src="https://github.com/MarkPrecursor/EEDS-keras/blob/master/EEDS.png" width="800"/>
</p>

My implementation has not really finished yet, but there still have something to see

## Use:
### Create your own data
open **prepare_data.py** and change the data path to your data

Excute:
`python prepare_data.py`

### training and test:
Excute:
`python main.py`


## Result:

EEDS and EES training for 100 epoches, with upscaling factor 2

|Method:| Bicubic | SRCNN | EEDS | EES |
|-------|---------|-------|------|-----|
|PSNR: |24.6971375057|28.6588428245|29.6439691166|28.4408566833|

Origin Image:
<p align="left">
  <img src="https://github.com/MarkPrecursor/EEDS-keras/blob/master/butterfly_GT.bmp" width="200"/>
</p>

input:
<p align="left">
  <img src="https://github.com/MarkPrecursor/EEDS-keras/blob/master/input.jpg" width="100"/>
</p>

EES:
<p align="left">
  <img src="https://github.com/MarkPrecursor/EEDS-keras/blob/master/EES_pre_adam100.jpg" width="200"/>
</p>

EEDS:
<p align="left">
  <img src="https://github.com/MarkPrecursor/EEDS-keras/blob/master/EEDS4_adam100.jpg" width="200"/>
</p>


## feature map visualization of EES:
After conv1:
<p align="left">
  <img src="https://github.com/MarkPrecursor/EEDS-keras/blob/master/layer1.png" width="800"/>
</p>

After Deconv:
<p align="left">
  <img src="https://github.com/MarkPrecursor/EEDS-keras/blob/master/layer1.png" width="800"/>
</p>

This code of feature map visualization part refered the code of https://www.kaggle.com/abhmul/leaf-classification/keras-convnet-lb-0-0052-w-visualization
