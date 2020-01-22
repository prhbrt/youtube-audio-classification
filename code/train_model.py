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

# normalize spectogram output
slope = K.variable(value=1/40)
intercept = K.variable(value=1)

model = Sequential()
model.add(Spectrogram(n_dft=512, n_hop=256, input_shape=(1, seconds), 
          return_decibel_spectrogram=True, power_spectrogram=2.0, 
          trainable_kernel=False, name='static_stft'))
model.add(Lambda(lambda x: slope * x + intercept))
model.add(Conv2D(32, (7, 7), name='conv1', activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPool2D((25, 17)))
model.add(Dropout(0.5))
model.add(Conv2D(32, (10, 10), name='conv2', activation='relu'))
model.add(Dropout(0.5))

# model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary(line_length=80, positions=[.33, .65, .8, 1.])


with open(f'{folder}/test_data.p3', 'rb') as f:
    X_test, y_test = pickle.load(f)


lr = 0.0003
optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)


for epoch in range(100):
    with open(f'{folder}/train_data_{epoch % 20}.p3', 'rb') as f:
        X_train, y_train = pickle.load(f)

    perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[perm, :, :]
    y_train = y_train[perm]

    history = model.fit(
        X_train, y_train, 
        batch_size=64, epochs=1, 
        validation_data=(X_test, y_test)
    )
    model.save(f'{folder}/model-2020-01-21-epoch-{epoch+1}.hd5')
    
#    with open(f'{folder}/model-2020-01-21-epoch-{epoch+1}.p3', 'wb') as f:
#        pickle.dump(model, f)
