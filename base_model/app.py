import model  # from model.py

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras

# import matplotlib.pyplot as plt
# %matplotlib inline

# load data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '../../ddd_images_train/drowsiness-scale',
    target_size=(128, 128),
    batch_size=32,
    shuffle=True,
    subset='training',
    class_mode='binary')

print(train_generator)

validation_generator = train_datagen.flow_from_directory(
    '../../ddd_images_train/drowsiness-scale',
    target_size=(128, 128),
    batch_size=32,
    shuffle=True,
    subset='validation',
    class_mode='binary')


test_generator = test_datagen.flow_from_directory(
    '../../ddd_images_test/drowsiness-scale',
    target_size=(128, 128),
    batch_size=32,
    shuffle=True,
    class_mode='binary')


# get model from model.py
baseModel = model.getModel()


# Train the model
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5',
                                       save_weights_only=True,
                                       monitor='val_accuracy'),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', update_freq='batch', profile_batch=0)
]

# The model weights (that are considered the best) are loaded into the model.
# model.load_weights(checkpoint_filepath)


# Train the model
hist = baseModel.fit(train_generator,
                     epochs=20,
                     batch_size=train_generator.samples,
                     validation_data=validation_generator,
                     validation_steps=validation_generator.samples,
                     workers=32,
                     verbose=1,
                     callbacks=my_callbacks)

# test
# score = baseModel.evaluate(test_generator,verbose=1)
# print('Test accuracy:', score[1])
