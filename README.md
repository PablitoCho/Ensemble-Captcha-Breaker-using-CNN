# Ensemble-Captcha-Breaker-using-CNN
6 digit captcha breaker using convolutional neural network. The final accuracy is 99%+ 

### Dependencies
|Name|Version|
|----|----|
|numpy|1.14.3|
|pandas|0.23.0|
|matplotlib|2.2.2|
|h5py|2.7.1|
|tensorflow|1.8.0|
|Keras|2.2.0|
|Pillow|5.1.0|

## Dataset Description
### Captcha Images used in training
<img src="https://i.imgur.com/mKQCi0F.png" width="240" height="80" /> <img src="https://i.imgur.com/V3nH1R6.png" width="240" height="80" /> <img src="https://i.imgur.com/4uxqvPw.png" width="240" height="80" />

### Labels in csv file
![label](https://i.imgur.com/YVSuZwL.jpg)

## Data Preprocessing
### EDA
 Given captcha images have 4 channels since it's RGBA. But as you can see above images, it looks monochrome ones, or at least colors do not matter obviously. So eliminating redundant channels has to be done before training so that we can save learning time.
 
 To do so, I needed to plot each channel and examine it with my own eyes.
```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
img = np.array(Image.open('./data/train/1.jpg'))
img.shape # (40, 120, 4) : (height, width, # of channels)
```
 First, see the original image.
```python
 # original image
imgplot = plt.imshow(img)
plt.show()
```
<img src="https://i.imgur.com/EGFq2z9.png" width="240" height="80" /> 

 Channel 1st
```python
imgplot = plt.imshow(img[:,:,0])
plt.show()
```
<img src="https://i.imgur.com/U4ZbrhX.png" width="240" height="80" /> 

 Channel 2nd
<img src="https://i.imgur.com/90o8nEd.png" width="240" height="80" /> 
Channel 3rd
<img src="https://i.imgur.com/waknbL6.png" width="240" height="80" /> 

 Actually, channels 1 to 3 have exactly same pixel values and based on what It looks, I guess they are all alpha channel for transparency.(I'm not that sure.)
 And they don't have proper digit information as you can see.

 Channel 4th
<img src="https://i.imgur.com/zSSszHa.png" width="240" height="80" /> 

 4th channel is what we have been looking forward to. I will use only this 4th channel, therefore, the images are put in the model as 1-channel ones.

## Model Architecture
### Visualization
![model](https://i.imgur.com/y1ASzGN.jpg)
### Details
```python
model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 40, 120, 1)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 40, 120, 32)       320       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 40, 120, 32)       9248      
_________________________________________________________________
batch_normalization_1 (Batch (None, 40, 120, 32)       128       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 20, 60, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 20, 60, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 20, 60, 64)        18496     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 20, 60, 64)        36928     
_________________________________________________________________
batch_normalization_2 (Batch (None, 20, 60, 64)        256       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 10, 30, 64)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 10, 30, 64)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 10, 30, 128)       73856     
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 10, 30, 128)       147584    
_________________________________________________________________
batch_normalization_3 (Batch (None, 10, 30, 128)       512       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 5, 15, 128)        0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 5, 15, 128)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 5, 15, 256)        295168    
_________________________________________________________________
batch_normalization_4 (Batch (None, 5, 15, 256)        1024      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 2, 7, 256)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 3584)              0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 3584)              0         
_________________________________________________________________
digit1 (Dense)               (None, 10)                35850     
=================================================================
Total params: 619,370
Trainable params: 618,410
Non-trainable params: 960
_________________________________________________________________
```

## Usages
### Codes Description
|Name|Description|
|----|----|
|img_preprocess.py| Turn jpg files into proper numpy array. |
|train_single_digit.py| Train each model for each digit. |
|ensemble_predictor.ipynb| Final ensemble model and Test it. |

## Results



## References
https://github.com/JackonYang/captcha-tensorflow

https://github.com/JasonLiTW/simple-railway-captcha-solver#english-version
