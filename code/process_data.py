import pickle
import os.path
import librosa
import numpy as np
from random import shuffle


from tqdm.auto import tqdm as ProgressBar

import tensorflow as tf

from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Lambda, Dropout
from keras.optimizers import Adam

from kapre.time_frequency import Spectrogram

from sklearn.model_selection import train_test_split


sampling_rate = 22050
seconds = sampling_rate * 5
folder = '/data/p253591/youtube_classification/data2/'

classes = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

data = [
    (label, os.path.join(folder, class_, filename))
    for label, class_ in enumerate(classes)
    for filename in os.listdir(os.path.join(folder, class_))
    if filename.endswith('.wav')
]
train_data, test_data = train_test_split(data, test_size=0.01)

X_test = []
y_test = []
for label, filename in ProgressBar(test_data):
    try:
        audio, sr = librosa.load(filename)
    except:
        pass
    leave = audio.shape[0] // seconds
    if sr != sampling_rate or leave <= 0:
        continue
    X_test.append(audio[:leave * seconds].reshape(leave, seconds))
    y_test.extend([label] * leave)

X_test = np.concatenate(X_test, axis=0)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
y_test = np.array(y_test)
with open(f'{folder}/test_data.p3', 'wb') as f:
    pickle.dump((X_test, y_test), f)


n = 20
for j in range(n):
    X_train = []
    y_train = []
    for i, (label, filename) in enumerate(ProgressBar(train_data[j::n])):
        try:
            audio, sr = librosa.load(filename)
        except:
            pass
        leave = audio.shape[0] // seconds
        if sr != sampling_rate or leave <= 0:
            continue
        X_train.append(audio[:leave * seconds].reshape(leave, seconds))
        y_train.extend([label] * leave)

    X_train = np.concatenate(X_train, axis=0)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    y_train = np.array(y_train)
    
    with open(f'{folder}/train_data_{j}.p3', 'wb') as f:
        pickle.dump((X_train, y_train), f)

