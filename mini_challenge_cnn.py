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

# ou X = dado.iloc[:,:1].values
#    y = dado.iloc[:,:2].values

quantity= sum(os.path.isfile(os.path.join(config.imagePath, f)) for f in os.listdir(config.imagePath))

for item in range(quantity):

    X.append(dado[item][0])
    y.append(dado[item][1])


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


train_set = train_datagen.flow_from_directory('/home/medeiros/mini_challenge/data_folder',
                                             target_size = (256, 256),
                                             color_mode = 'rgb',
                                             classes = ['graduation', 'meeting', 'picnic'],
                                             batch_size = 32,
                                             class_mode = 'categorical') #Recebe um diretório e caminha por ele
                                                                        #com base nos parâmetros fornecidos


test_set = test_datagen.flow_from_directory('/home/medeiros/mini_challenge/data_folder',
                                           target_size = (256, 256),
                                           color_mode = 'rgb',
                                           classes = ['graduation', 'meeting', 'picnic'],
                                           batch_size = 32,
                                           class_mode = 'categorical')




model = Sequential()

model.add(Convolution2D(filters = 32,
                       kernel_size = (5, 5),
                       input_shape = (64, 64, 3),
                       activation = 'relu')) #Primeira camada convolucional com ativação ReLU
model.add(MaxPooling2D(pool_size = (2, 2),
                      strides = 2)) #Pooling Layer de passo 2 que divide a dimensão na metade

model.add(Convolution2D(filters = 32,
                       kernel_size = (5, 5),
                       input_shape = (32, 32, 3),
                       activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2),
                      strides = 2))

model.add(Convolution2D(filters = 64,
                       kernel_size = (5, 5),
                       input_shape = (16, 16, 3),
                       activation = 'relu'))

model.add(Flatten()) #Converte para uma dimensão

model.add(Dense(activation = 'relu',
               units = 16384)) #FC1
model.add(Dense(activation = 'relu',
               units = 1024)) #FC2
model.add(Dense(activation = 'softmax',
               units = 3))#FC3

print()


################################### VERSAO ANTERIOR #########################################

#Data Preprocessing
from config import Config

#Importing Labraries
import sys
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Importing the Keras libraries and packages
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


from PIL import Image

#Import the dataset
config = Config()

dado = pd.read_csv(config.arquivo_csv).values

imagem = dado[0][0]
categoria = dado[0][1]

X = []
y = []

# ou X = dado.iloc[:,:1].values
#    y = dado.iloc[:,:2].values

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

# Part 1 - Building the CNN