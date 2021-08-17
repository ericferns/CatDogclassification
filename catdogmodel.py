#import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/home/vf/icart_release_cpu/YOLO/products/model_creation/train/train', #change this directory to ./train
                                                 target_size = (64, 64),
                                                 batch_size = 5,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('C:/home/vf/icart_release_cpu/YOLO/products/model_creation/train/test1', #change this directory ./test1
                                            target_size = (64, 64),
                                            batch_size = 5,
                                            class_mode = 'binary')

classifier=Sequential()
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit_generator(training_set,samples_per_epoch=8000,nb_epoch=4,validation_data=test_set,nb_val_samples=2000)

model_json = classifier.to_json()
with open("./model.json","w") as json_file:
  json_file.write(model_json)

# Step 13: Save the weights in a seperate file
classifier.save_weights("./model.h5")

print("Model trained Successfully!")
