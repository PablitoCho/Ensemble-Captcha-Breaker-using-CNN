# Ensemble-Captcha-Breaker-using-CNN
6 digit captcha breaker using convolutional neural network. The final accuracy is 99%+ 

## Dataset Description
### Captcha Images used in training
<img src="https://i.imgur.com/mKQCi0F.png" width="240" height="80" /> <img src="https://i.imgur.com/V3nH1R6.png" width="240" height="80" /> <img src="https://i.imgur.com/4uxqvPw.png" width="240" height="80" />

### Labels in csv file
![label](https://i.imgur.com/YVSuZwL.jpg)

## Data Preprocessing

## Model Architecture
### Visualization
![model](https://i.imgur.com/y1ASzGN.jpg)

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


## Usages

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
