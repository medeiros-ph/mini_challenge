# Importing the Keras libraries and packages

import keras
import sys
import os
import os.path
import time
import timeit
from keras import backend as K

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.core import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from keras.utils import to_categorical
from keras.utils import to_categorical
from keras.utils import to_categorical

from subprocess import check_output
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from config import Config
from sklearn.model_selection import train_test_split
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

from PIL import Image

config = Config()

dado = pd.read_csv(config.arquivo_csv).values

#quantity = sum(os.path.isfile(os.path.join(config.imagePath, f)) for f in os.listdir(config.imagePath))

#for item in range(quantity):
#    X.append(dado[item][0])
#    y.append(dado[item][1])

#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


model = Sequential()

model.add(Convolution2D(filters = 32,
                       kernel_size = (5, 5),
                       input_shape = (64, 64, 3),
                       activation = 'relu')) #Primeira camada convolucional com ativação ReLU
model.add(MaxPooling2D(pool_size = (2, 2),
                      strides = 2)) #Pooling Layer de passo 2 que divide a dimensão na metade

model.add(Convolution2D(filters = 32,
                       kernel_size = (5, 5),
                       activation = 'relu'))
#input_shape = (32, 32, 3),
model.add(MaxPooling2D(pool_size = (2,2),
                      strides = 2))

model.add(Convolution2D(filters = 64,
                       kernel_size = (5, 5),
                       activation = 'relu'))
#input_shape = (16, 16, 3),

model.add(Flatten()) #Converte para uma dimensão

# Dense = Full connection
model.add(Dense(activation = 'relu',
               units = 16384)) #FC1
model.add(Dense(activation = 'relu',
               units = 1024)) #FC2
model.add(Dense(activation = 'softmax',
               units = 3))#FC3

#compila a arquitetura
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None)

#Data Augumentation
#old train_datagen = gen
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(directory=r"/home/medeiros/mini_challenge/data_folder/treino",
                                             target_size = (64, 64),
                                             color_mode = 'rgb',
                                             classes = ['graduation', 'meeting', 'picnic'],
                                             batch_size = 32,
                                             class_mode = 'categorical',
                                             shuffle=True,
                                             seed=42
 ) #Recebe um diretório e caminha por ele
   #com base nos parâmetros fornecidos

#print(train_generator)

test_set = test_datagen.flow_from_directory(directory=r"/home/medeiros/mini_challenge/data_folder/teste",
                                           target_size = (64, 64),
                                           color_mode = 'rgb',
                                           classes = ['graduation', 'meeting', 'picnic'],
                                           batch_size = 32,
                                           class_mode = 'categorical')


validation_generator = test_datagen.flow_from_directory(directory=r"/home/medeiros/mini_challenge/data_folder/validation",
                                           target_size = (64, 64),
                                           color_mode = 'rgb',
                                           classes = ['graduation', 'meeting', 'picnic'],
                                           batch_size = 32,
                                           class_mode = 'categorical')

def get_callbacks(name_weights, patience_lr, name_csv):
    mcp_save = ModelCheckpoint(name_weights)
    csv_logger = CSVLogger(name_csv)
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='max')
    return [mcp_save, csv_logger, reduce_lr_loss]

# y_train = to_categorical(y_train)
# y_val = to_categorical(y_val)
# y_test = to_categorical(y_test)

batch_size=32
start = time.perf_counter()

# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
# The first is the allow_growth option, which attempts to allocate only as much
# GPU memory based on runtime allocations: it starts out allocating very little 
# memory, and as Sessions get run and more GPU memory is needed, we extend the GPU memory region needed by 
# the TensorFlow process. Note that we do not release memory, since that can lead 
# to even worse memory fragmentation. To turn this option on, set the option in the ConfigProto by:

config = tf.ConfigProto()
config.gpu_options.allow_growth = False

sess = tf.Session(config=config)


name_weights = "Train_model_weights_{epoch:02d}.h5"
csv_name = "trainning_k.csv"
callbacks = get_callbacks(name_weights = name_weights, patience_lr=10, name_csv = csv_name)
#train_generator = train_datagen.flow(x_train, y_train, batch_size = batch_size)
#generator
#early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')

history = model.fit_generator(
        train_generator,
        epochs = 50,
        steps_per_epoch = 12592,
        verbose = 1,
        validation_data = validation_generator,
        validation_steps = 4179,
        callbacks = callbacks)

#epochs = 50,
#steps_per_epoch = 12592,
#verbose = 1,
#callbacks = [early_stopping],
#validation_data = validation_generator,
#validation_steps = 4179)



RefTitleResults = 'L2D3D3G0B1E200_3003001'

###### Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy ')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
#plt.show()
plt.savefig("AccxEpoch" + RefTitleResults + ".png")

###### Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss ')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
#plt.show()
plt.savefig("LossxEpoch" + RefTitleResults + ".png")

sess.close()

elapsed = time.perf_counter() - start
print('Elapsed %.3f seconds.' % elapsed)    

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# # print(model.evaluate(x_test, y_test))
# #print(x_test.shape)
# probs = model.predict(validation_generator)
# ######## keep probabilities for the positive outcome only
# probsp = probs  # [:, 1]
# # print(probsp.shape)
# # print(y_val)
# # print(probs)
# ######## calculate AUC
# auc = roc_auc_score(validation_generator, probsp)
# print('AUC: %.3f' % auc)

# ######## calculate roc curve
# fpr, tpr, thresholds = roc_curve(validation_generator, probsp)

# plt.figure()
# plt.plot([0, 1], [0, 1], 'k--')  # k = color black
# plt.plot(fpr, tpr, label="AUC: %.3f" % auc, color='C1', linewidth=3)  # for color 'C'+str(j), for j[0 9]
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
# plt.title('ROC')
# plt.xlabel('false positive rate', fontsize=14)
# plt.ylabel('true positive rate', fontsize=14)

# #plt.show()
# plt.savefig("ROCLensDetectNet" + RefTitleResults + ".png")

# save model

model.save('Model' + RefTitleResults + '.h5')

