import numpy as np
#import matplotlib.pyplot as plt
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