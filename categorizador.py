import csv
import os
import fileinput
import shutil
import numpy
from random import randint, shuffle

locations = "/home/medeiros/mini_challenge/dataset_images_minitest.csv"

with open(locations, 'r') as f:
    rows = csv.reader(f, delimiter=',')
    next(rows)
    for row in rows:
        file_sel = row[0] # files names on first column
        print(file_sel)
        folder_sel = row[1] # directory names on second column

        # create directory if it does not exist
        if (folder_sel not in list(os.walk('.'))[0][1]):
            os.mkdir(folder_sel)
	
        # if file does not exist go to the next one
        if (file_sel not in list(os.walk('.'))[0][2]):
            continue
	
        # move file
        new_file_path = os.path.join(folder_sel, file_sel)
        os.rename(file_sel, new_file_path)

# create directory if it does not exist
if ('treino' not in list(os.walk('.'))[0][1]):
	os.mkdir('treino')
        
# create directory if it does not exist
if ('teste' not in list(os.walk('.'))[0][1]):
	os.mkdir('teste')

# create directory if it does not exist
if ('validation' not in list(os.walk('.'))[0][1]):
	os.mkdir('validation')
	
####

print("Parte 2: pastas treino e teste criadas")
	
FF=('graduation','meeting','picnic')
	
for name in FF:
	if (name not in list(os.walk('teste'))[0][1]):
		os.mkdir(os.path.join('teste', name))	
		
	if (name not in list(os.walk('teste'))[0][1]):
		os.mkdir(os.path.join('train', name))

	if (name not in list(os.walk('validation'))[0][1]):
		os.mkdir(os.path.join('validation', name))

for name in FF: # for each category
	imgs = os.listdir(name)
	shuffle(imgs)
	l_size = len(imgs)
	first_mark = int(l_size * 0.6)
	second_mark = int(l_size * (0.6 + 0.2))
	train_files = imgs[:first_mark]
	test_files = imgs[first_mark:second_mark]
	validation_files = imgs[second_mark:]
	print("l_size (60%): {}".format(l_size))
	
		
	### train
	if name not in os.listdir('treino'):
		os.mkdir(os.path.join('treino', name))
	for f in train_files:
		# move files
		#os.rename(os.path.join(name, f), os.path.join('treino', name, f))
		# copy files
		shutil.copy(os.path.join(name, f), os.path.join('treino', name, f))
	
	### test
	if name not in os.listdir('teste'):
		os.mkdir(os.path.join('teste', name))
	for f in test_files:
		# move files
		#os.rename(os.path.join(name, f), os.path.join('teste', name, f))
		# copy files
		shutil.copy(os.path.join(name, f), os.path.join('teste', name, f))

	### validation
	if name not in os.listdir('validation'):
		os.mkdir(os.path.join('validation', name))
	for f in test_files:
		# move files
		#os.rename(os.path.join(name, f), os.path.join('validation', name, f))
		# copy files
		shutil.copy(os.path.join(name, f), os.path.join('validation', name, f))	


