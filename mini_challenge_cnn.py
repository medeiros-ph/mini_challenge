#Classes

# graduation #
# picnic #
# meeting #



# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.core import Dropout
#from keras.la
from keras import backend as K



# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(filters = 32, kernel_size = (3,3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

## config net until here

# Compiling the CNN
#definir os paramentros de aprendizagem da rede
x = classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print("[INFO] {}".format(x))

# Part 2 - Fitting the CNN to the images
################### Processamento de Imagens ########################
from keras.preprocessing.image import ImageDataGenerator

#classe
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 classes=['dogs', 'cats'],
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            classes=['dogs', 'cats'],
                                            batch_size = 32,
                                            class_mode = 'binary')
######################################################################


classifier.fit_generator(training_set,
                         steps_per_epoch= 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)