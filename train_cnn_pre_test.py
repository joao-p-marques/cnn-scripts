
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from keras.utils import to_categorical

from keras.applications import VGG16

# create a new generator
imagegen = ImageDataGenerator()
# load train data
train = imagegen.flow_from_directory("datasets/imagenette2/train/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))
# load val data
val = imagegen.flow_from_directory("datasets/imagenette2/val/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))

# include top should be False to remove the softmax layer
pretrained_model = VGG16(include_top=False, weights='imagenet')
pretrained_model.summary()

# extract train and val features
print("Importing train features...")
vgg_features_train = pretrained_model.predict_generator(train)
print("Importing test features...")
vgg_features_val = pretrained_model.predict(val)

# OHE target column
train_target = to_categorical(train.labels)
val_target = to_categorical(val.labels)

model2 = Sequential()
model2.add(Flatten(input_shape=(7,7,512)))
model2.add(Dense(100, activation='relu'))
model2.add(Dropout(0.5))
model2.add(BatchNormalization())
model2.add(Dense(10, activation='softmax'))

# compile the model
model2.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

model2.summary()

# train model using features generated from VGG16 model
model2.fit(vgg_features_train, train_target, epochs=50, batch_size=128, validation_data=(vgg_features_val, val_target))
