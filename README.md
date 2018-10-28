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

### Codes Description
|Name|Description|
|----|----|
|img_preprocess.py| Turn jpg files into proper numpy array. |
|train_single_digit.py| Train each model for each digit. |
|ensemble_predictor.ipynb| Final ensemble model and Test it. |

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

 So below is the final data preprocessing code. You can see I choose only 4th channel([:,:,3]) and normalize by dividing with 255.0.

 [img_preprocess.py]
 ```python
from PIL import Image
import numpy as np
import pandas as pd
import os
LETTERSTR = "0123456789" # domain for captcha letters, all 10 digits
NUM_TRAIN = # of train images you prepared.
NUM_VAL = # of validation set

# parse label information
def to_one_hot(label, digit):
    onehot = [0 for i in range(len(LETTERSTR))]
    onehot[LETTERSTR.find(label[digit])] = 1
    return onehot

train_csv = pd.read_csv('captcha_training_label.csv')
train_csv['label_str'] = [str(n).zfill(6) for n in train_csv['label']] # (NUM_TRAIN, 3)
val_csv = pd.read_csv('captcha_validation_label.csv')
val_csv['label_str'] = [str(n).zfill(6) for n in val_csv['label']] # (NUM_VAL, 3)

train_label = []
val_label = []
for digit in range(6): # 6 is the number of digit in captcha
    train_label.append(np.array([to_one_hot(label_str, digit=digit) for label_str in train_csv['label_str']], dtype='int32'))
    val_label.append(np.array([to_one_hot(label_str, digit=digit) for label_str in val_csv['label_str']], dtype='int32'))
    
train_data = np.stack([(np.array(Image.open('./data/train/' + str(i) + ".jpg"))/255.0)[:,:,3] for i in range(1, NUM_TRAIN+1)])
train_data = train_data[:,:,:,np.newaxis]
# shape of train_data : (NUM_TRAIN, 40, 120, 1) - (# of data, height of img, width of img, channel).
val_data = np.stack([(np.array(Image.open('./data/val/' + str(i) + ".jpg"))/255.0)[:,:,3] for i in range(1, NUM_VAL+1)])
val_data = val_data[:,:,:,np.newaxis]
# shape of val_data : (NUM_VAL, 40, 120, 1) 
 ```

## Model Architecture
 I used wonderful work from [https://github.com/JasonLiTW/simple-railway-captcha-solver], of course modified a littie the original model architecture. But after couple of times training it, I found my model did not go well. Each single digit accuracy was 90%+, but combining all 6 accuracy just around 70%. From the original work, one model predicted all 6 digits and the training went based on 6th digit accuracy. And I took notice that 6th digit accuracy was almost 100% but others are around 90% more or less.
 
 And I thought that if I trained 6 models for 6 digits seperately and combined them together like ensemble model. The final accuracy would increase significantly. And this idea is the key point over my job here.

 Each model looks like below.

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


### Usages

 This is the code to train model for each digit.

 [train_single_digit.py]
```python
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('tf')
from img_preprocess import train_data, val_data, train_label, val_label, LETTERSTR
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--digit", required=True, help="the index of digit in captcha image to predict. From 1 to 6.")
ap.add_argument("--trial", required=False, default=1, help="trial number. change when the model is tuned and modified.")
args = vars(ap.parse_args())

NUM_digit = int(args['digit'])
NUM_trial = int(args['trial'])

outnode = 'digit' + str(NUM_digit)

train_label_single_digit = train_label[NUM_digit-1]
val_label_single_digit = val_label[NUM_digit-1]

#def model(input_shape, num_targets, num_digits, layers, dropout):
def model(input_shape, layers, dropout):
    in_ = Input(input_shape)
    out = in_
    
    out = Conv2D(filters=layers[0], kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = Conv2D(filters=layers[0], kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Dropout(dropout)(out)
    
    out = Conv2D(filters=layers[1], kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = Conv2D(filters=layers[1], kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Dropout(dropout)(out)
    
    out = Conv2D(filters=layers[2], kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = Conv2D(filters=layers[2], kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Dropout(dropout)(out)
    
    out = Conv2D(filters=layers[3], kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    
    out = Flatten()(out)
    out = Dropout(dropout)(out)
    #out = [Dense(num_targets, name='digit' + str(i+1), activation='softmax')(out) for i in range(num_digits)]
    out = Dense(10, name=outnode, activation='softmax')(out)
    model = Model(inputs=in_, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model
	
#captcha_breaker = model(input_shape=train_data.shape[1:], num_targets=10,  num_digits=6, layers=[32, 64, 128, 256], dropout=0.3)
captcha_breaker = model(input_shape=train_data.shape[1:], layers=[32, 64, 128, 256], dropout=0.3)

#captcha_breaker.summary()
# for advanced and easy-to-reproduce process
filepath="digit" + str(NUM_digit) + "_weights[" + str(NUM_trial) + "].h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor = 'val_acc', patience=5, verbose=1, mode='auto')
callbacks_list = [checkpoint, earlystop]
captcha_breaker.fit(train_data, train_label_single_digit, batch_size=100, epochs=100, verbose=2, validation_data=(val_data, val_label_single_digit), callbacks=callbacks_list)
# save the model architecture
with open('captcha_digit' + str(NUM_digit) + ' _breaker[' + str(NUM_trial) + '].json', 'w') as f:
    f.write(captcha_breaker.to_json())

print('done!')
```

 This is a command in console.

```python
python train_single_digit.py --digit 1 --trial 1
```

 Above command means that training model for 1st digit with 1st trial. Trial number has its default as 1. When you change the model architecture significantly, adjust the number.
 
 As results, after long training... `digit1_weight[1].h5` file for weight values and `captcha_digit1_breaker[1].json` file for model structure would be generated. We need 6 hdf files since there are 6 digits in our target captcha images.

## Results

 Every accuracy for each digit above 99% and some have 100%. Acutally, I used validation set with only 2000 images(it was really really hard to label all the downloaded captcha images mannually.

 And overall accuracy was above 99.5%.

## References
https://github.com/JackonYang/captcha-tensorflow

https://github.com/JasonLiTW/simple-railway-captcha-solver
