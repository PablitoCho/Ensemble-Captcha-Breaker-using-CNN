# Ensemble-Captcha-Breaker-using-CNN
6 digit captcha breaker using convolutional neural network. The final accuracy is 99%+ 

## Dataset Description
### Captcha Images used in training
<img src="https://i.imgur.com/mKQCi0F.png" width="240" height="80" /> <img src="https://i.imgur.com/V3nH1R6.png" width="240" height="80" /> <img src="https://i.imgur.com/4uxqvPw.png" width="240" height="80" />

### Labels in csv file
![label](https://i.imgur.com/YVSuZwL.jpg)

## Data Preprocessing

## Model Architecture
![model](https://i.imgur.com/y1ASzGN.jpg)

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
