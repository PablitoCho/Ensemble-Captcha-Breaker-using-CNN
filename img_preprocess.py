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
# We can just use 4th one only. I guess...
val_data = np.stack([(np.array(Image.open('./data/val/' + str(i) + ".jpg"))/255.0)[:,:,3] for i in range(1, NUM_VAL+1)])
val_data = val_data[:,:,:,np.newaxis]
# shape of val_data : (NUM_VAL, 40, 120, 1)
