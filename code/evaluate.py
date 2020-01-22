import pickle
import os.path
import librosa
import numpy
from random import shuffle

from tqdm.auto import tqdm as ProgressBar

import tensorflow as tf

from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Lambda, Dropout
from keras.optimizers import Adam

from kapre.time_frequency import Spectrogram
from skimage.io import imsave

from sklearn.metrics import classification_report


sampling_rate = 22050
seconds = sampling_rate * 5
folder = '/data/p253591/youtube_classification/data2/'


with open(f'{folder}/test_data.p3', 'rb') as f:
    X_test, y_test = pickle.load(f)

# normalize spectogram output
slope = K.variable(value=1/40)
intercept = K.variable(value=1)

spectogram_model = Sequential()
spectogram_model.add(Spectrogram(n_dft=512, n_hop=256, input_shape=(1, seconds), 
          return_decibel_spectrogram=True, power_spectrogram=2.0, 
          trainable_kernel=False, name='static_stft'))
spectogram_model.add(Lambda(lambda x: slope * x + intercept))

for sample in numpy.random.permutation(len(X_test))[:4]:
    y_out = spectogram_model.predict(X_test[sample:sample+1])

    im = (y_out[0] + 1) / 2.0
    im = im * [[[1, 0, 0]]] + (1-im) * [[[0, 0, 1]]]
    imsave(f'spectrogram-{sample}-label-{y_test[sample]}.png', im)


model = load_model(f'{folder}/model-2020-01-21-epoch-21.hd5', custom_objects={'Spectrogram': Spectrogram, 'slope': slope, 'intercept': intercept})
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred[:, 0] > 0.5, target_names=['RTV Oost', 'RTV Noord']))
