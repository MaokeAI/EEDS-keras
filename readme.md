# Keras implementation of EEDS: End-to-End image super-resolution via deep and shallow convolutional networks


The original paper is [end-to-end image super-resolution via deep and shallow convolutional networks](https://arxiv.org/abs/1607.07680)

<p align="center">
  <img src="https://github.com/MarkPrecursor/EEDS-keras/blob/master/EEDS.png" width="800"/>
</p>

This implementation only focuse on two-scales' upsampling, and it is easy to change this code to perform upscaling with other ratio by change the strides in **Conv2DTranspose**. And because of the difference between the color space of YCrCb in Matlab and OpenCV, the final PSNR will have some difference. Results evluated with the MATLAB code will be a little higher. 

## Run the demo:
Excute:
`python EEDS_predict_demo.py`

## Train your own data:
### Create your own data
open **prepare_data.py** and change the data path to your data

Excute:
`python prepare_data.py`

### training EEDS and test:
Excute:
`python EEDS.py`

### training EES and test:
Excute:
`python EES.py`


## Result:

EEDS and EES training for 200 epoches, with upscaling factor 2
<p align="left">
  <img src="https://github.com/MarkPrecursor/SRCNN-keras/blob/master/result.jpg" width="800"/>
</p>

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
  <img src="https://github.com/MarkPrecursor/EEDS-keras/blob/master/layer2.png" width="800"/>
</p>

This code of feature map visualization part refered the code of https://www.kaggle.com/abhmul/leaf-classification/keras-convnet-lb-0-0052-w-visualization
