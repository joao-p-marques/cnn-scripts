
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import collections

from predict import predict, predict_remote_image
from preprocessing import split_files, map_classes

print('imports success')

base_dir = './food-5/'

split_files(base_dir)
map_classes(base_dir)

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# each class will be inside here

# all images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1.0/255.)
test_datagen = ImageDataGenerator(rescale=1.0/255.)

# flow train images in batches of 20 using train_datagen
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=16,
                                                    class_mode='categorical',
                                                    shuffle=False,
                                                    target_size=(244, 244))

# flow validation images in batches of 20 using train_datagen 
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  batch_size=16,
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  target_size=(244, 244))

# import pretrained model
pretrained_model = VGG16(weights = 'imagenet',include_top = False)

pretrained_model.summary()

# extract train and val features
print('Extracting train features...')
pretrained_features_train = pretrained_model.predict(train_generator)
print('Extracting test features...')
pretrained_features_test = pretrained_model.predict(test_generator)

# OHE target column
train_target = to_categorical(train_generator.labels)
test_target = to_categorical(test_generator.labels)

model2 = Sequential()
model2.add(Flatten(input_shape=(7, 7, 512)))
model2.add(Dense(100, activation='relu'))
model2.add(Dropout(0.5))
model2.add(BatchNormalization())
class_num = 5
model2.add(Dense(class_nuclass_num, activation='softmax'))

# compile the model
model2.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

model2.summary()

checkpointer = ModelCheckpoint(filepath='model.3.hdf5', verbose=1, save_best_only=True)

# train model using features generated from VGG16 model
model2.fit(pretrained_features_train, train_target, epochs=50, batch_size=16, validation_data=(pretrained_features_test, test_target), callbacks=[checkpointer])

predict_remote_image(url='https://lmld.org/wp-content/uploads/2012/07/Chocolate-Ice-Cream-3.jpg', model=model, ix_to_class=ix_to_class, debug=True)
predict_remote_image(url='https://images-gmi-pmc.edge-generalmills.com/75593ed5-420b-4782-8eae-56bdfbc2586b.jpg', model=model, ix_to_class=ix_to_class, debug=True)
