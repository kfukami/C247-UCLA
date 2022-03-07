# lstm.py
# Kai Fukami, UCLA MAE
# Ver1: 2022/03/06

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Add, Dense, Conv2D,Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, Reshape, LSTM, Dropout, BatchNormalization,Permute,GRU
from keras.models import Model, Sequential
from keras.models import load_model
from keras import backend as K
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from keras import regularizers
from keras.utils import to_categorical


import tensorflow as tf
from keras.backend import tensorflow_backend
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        allow_growth=True,
        visible_device_list="2"
    )
)
session = tf.Session(config=config)
tensorflow_backend.set_session(session)



## Loading the numpy arrays corresponding to the EEG dataset
X_test = np.load("../X_test.npy")
y_test = np.load("../y_test.npy")
person_train_valid = np.load("../person_train_valid.npy")
X_train_valid = np.load("../X_train_valid.npy")
y_train_valid = np.load("../y_train_valid.npy")
person_test = np.load("../person_test.npy")

## Printing the shapes of the numpy arrays

print ('Training/Valid data shape: {}'.format(X_train_valid.shape))
print ('Test data shape: {}'.format(X_test.shape))
print ('Training/Valid target shape: {}'.format(y_train_valid.shape))
print ('Test target shape: {}'.format(y_test.shape))
print ('Person train/valid shape: {}'.format(person_train_valid.shape))
print ('Person test shape: {}'.format(person_test.shape))


y_train_valid -= 769
y_test -= 769

## Creating the training and validation sets

# First generating the training and validation indices using random splitting
ind_valid = np.random.choice(2115, 500, replace=False)
ind_train = np.array(list(set(range(2115)).difference(set(ind_valid))))

# Creating the training and validation sets using the generated indices
(x_train, x_valid) = X_train_valid[ind_train], X_train_valid[ind_valid] 
(y_train, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]
print('Shape of training set:',x_train.shape)
print('Shape of validation set:',x_valid.shape)
print('Shape of training labels:',y_train.shape)
print('Shape of validation labels:',y_valid.shape)


# Converting the labels to categorical variables for multiclass classification
y_train = to_categorical(y_train, 4)
y_valid = to_categorical(y_valid, 4)
y_test = to_categorical(y_test, 4)
print('Shape of training labels after categorical conversion:',y_train.shape)
print('Shape of validation labels after categorical conversion:',y_valid.shape)
print('Shape of test labels after categorical conversion:',y_test.shape)

# Adding width of the segment to be 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_valid.shape[2])
x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
print('Shape of training set after adding width info:',x_train.shape)
print('Shape of validation set after adding width info:',x_valid.shape)
print('Shape of test set after adding width info:',x_test.shape)


# # Reshaping the training and validation dataset
x_train = np.swapaxes(x_train, 1,2)
# x_train = np.swapaxes(x_train, 1,2)
# x_valid = np.swapaxes(x_valid, 1,3)
x_valid = np.swapaxes(x_valid, 1,2)
# x_test = np.swapaxes(x_test, 1,3)
x_test = np.swapaxes(x_test, 1,2)
print('Shape of training set after dimension reshaping:',x_train.shape)
print('Shape of validation set after dimension reshaping:',x_valid.shape)
print('Shape of test set after dimension reshaping:',x_test.shape)


num_time = 200

rate = 1.0e-3
act = 'tanh'
input_img = Input(shape=(num_time,22))
x = LSTM(128,return_sequences=True,activation=act)(input_img)
x = Dropout(0.25)(x)
x = Flatten()(x)
x_final = Dense(4, activation='softmax')(x)

model = Model(input_img, x_final)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


## Training
from keras.callbacks import ModelCheckpoint,EarlyStopping
model_cb = ModelCheckpoint('../model_box/lstm_nt200_cv3.hdf5', monitor='val_loss',save_best_only=True,verbose=1) 
#########################################################################################
###############################Early Stopping############################################
early_cb = EarlyStopping(monitor='val_loss', patience=100,verbose=1)
#########################################################################################
###############################make a model############################################
cb = [model_cb, early_cb]
history = model.fit(x_train[:,:num_time,:],y_train,epochs=5000,batch_size=256,verbose=1,callbacks=cb,shuffle=True,validation_data=(x_valid[:,:num_time,:], y_valid))
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
####################save a history curve##########################
df_results.to_csv(path_or_buf='../history_box/lstm_nt200_cv3.csv',index=False)


# x1 = Dense(256, activation=act)(input_img_1)
# x1 = Reshape([1,256])(x1)
# x1 = LSTM(256,return_sequences=True,activation=act,batch_input_shape=(None, None, 256))(x1)
# x_123 = Add()([x,x1,x2])

score = model.evaluate(x_test[:,:num_time,:], y_test, verbose=0)
print('Test accuracy:',score[1])

