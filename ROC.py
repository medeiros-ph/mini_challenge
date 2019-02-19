#Prediçoes e ROC

# a) Carregando modelo salvo
from keras.models import load_model

model = load_model('file.h5')


# b) Datagen para lote de predição
from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale = 1./255)

to_predict = test_datagen.flow_from_directory('dataset1/test',
                                           target_size = (64, 64),
                                           color_mode = 'rgb',
                                           classes = ['graduation', 'meeting', 'picnic'],
                                           batch_size = 1,
                                           class_mode = None,
                                           shuffle = False)

# c) Predições
to_predict.reset()

from keras.models import Sequential

predict = model.predict_generator(to_predict, verbose = 1)

#d) ROCurve
from sklearn.preprocessing import label_binarize

y = label_binarize(to_predict.classes, classes=[0, 1, 2])

#Binariza a classificação do "teste set" de acordo
#de acordo com as classes dadas



from scipy import interp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

classes_number = y.shape[1] #determina a quantidade de classes de acordo com o set binarizado
line = 2

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(classes_number):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], predict[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i]) #Calcula fpr, tpr e a roc para cada classe

fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), predict.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes_number)]))

mean_tpr = np.zeros_like(all_fpr)

for i in range(classes_number):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= classes_number

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

#plot
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(classes_number), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=line,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=line)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' Model - Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('rocurve.jpg')