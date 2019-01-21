

# Part 1 - Building the CNN
# Importing the Keras libraries and packages

import sys, os
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
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image



train = pd.read_csv("input/dataset_images_minitest (train).csv").values
test  = pd.read_csv("input/dataset_images_minitest (test).csv").values

print (train)

# Reshape and normalize training data


# Reshape and normalize test data

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()