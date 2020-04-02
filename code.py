
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import librosa
import glob 


# Importing the data set
file_name =
data,sampling_rate = librosa.load(file_name,res_type='kaiser_fast')
mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T,axis=0)

feature = mfccs
label = 
    
from sklearn.preprocessing import LabelEncoder

X = np.array(feature.tolist())
y = np.array(label.tolist())
lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(y))
num_labels = y.shape[1]


# Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

classifier = Sequential()

classifier.add(LSTM(units = 50, return_sequences = True, input_shape = ()))
classifier.add(Dropout(0.2))

classifier.add(LSTM(units = 50, return_sequences = True))
classifier.add(Dropout(0.2))

classifier.add(LSTM(units = 50))
classifier.add(Dropout(0.2))

classifier.add(Dense(num_labels))
model.add(Activation('softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics =['accuracy'])
classifier.fit(X, y, epochs = 100, batch_size = 10)