# Keras implementation of EEDS: End-to-End image super-resolution via deep and shallow convolutional networks


The original paper is [end-to-end image super-resolution via deep and shallow convolutional networks](https://arxiv.org/abs/1607.07680)

![EEDS]("./EEDS.png")

My implementation has not really finished yet, but there still have something to see

## Use:
### Create your own data
open **prepare_data.py** and change the data path to your data

Excute:
'python prepare_data.py'

### training and test:
Excute:
'python main.py'


## Result(EEDS training for 100 epoches and EES training for 200 epoches, with upscaling factor 2):

|Method:| Bicubic | SRCNN | EEDS | EES |
|-------|---------|-------|------|-----|
|PSNR: |24.6971375057|28.6588428245|29.6439691166|28.4408566833|

Origin Image:
![Origin]("./butterfly_GT.bmp")

input:
![input]("./input.jpg")

EES:
![EES]("./EES_adam200.jpg")

EEDS: 
![EEDS]("./EEDS4_adam100.jpg")



