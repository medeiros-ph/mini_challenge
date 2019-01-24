

# Part 1 - Building the CNN
# Importing the Keras libraries and packages

import sys
import os
import os.path
from keras import backend as K

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.core import Dropout

from subprocess import check_output
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from config import Config
from sklearn.model_selection import train_test_split
from PIL import Image

config = Config()

dado = pd.read_csv(config.arquivo_csv).values

imagem = dado[0][0]
categoria = dado[0][1]

X = []
y = []

quantity= sum(os.path.isfile(os.path.join(config.imagePath, f)) for f in os.listdir(config.imagePath))

for item in range(quantity):

    X.append(dado[item][0])
    y.append(dado[item][1])


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

print()



# Reshape and normalize training data


# Reshape and normalize test data

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()