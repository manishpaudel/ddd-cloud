# Import packages and set numpy random seed
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import matplotlib.pyplot as plt
# %matplotlib inline


#load data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '../../ddd_images_train/drowsiness-scale',
    target_size=(144, 144),
    batch_size=64,
    shuffle=True,
    subset='training',
    class_mode='binary')

print(train_generator)

validation_generator = train_datagen.flow_from_directory(
    '../../ddd_images_train/drowsiness-scale',
    target_size=(144, 144),
    batch_size=64,
    shuffle=True,
    subset='validation',
    class_mode='binary')


test_generator = test_datagen.flow_from_directory(
    '../../ddd_images_test/drowsiness-scale',
    target_size=(144, 144),
    batch_size=64,
    shuffle=True,
    class_mode='binary')

model = tf.keras.applications.ResNet50V2(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(144,144,3), pooling='max', classes=2,
    classifier_activation='sigmoid'
)
model.summary()

# Compile the model
model.compile(optimizer='adam', 
              loss=['binary_crossentropy'], 
              metrics=['accuracy'])


# logging and callbacks

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5', 
                                       save_weights_only=True,
                                       monitor='val_accuracy'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq='batch', profile_batch = 0),
]

# The model weights (that are considered the best) are loaded into the model.
# model.load_weights(checkpoint_filepath)

# Train the model
hist = model.fit(train_generator,
                 epochs=20,
                 batch_size=train_generator.samples,
                 validation_data=validation_generator,
                 validation_steps=validation_generator.samples,
                 callbacks = my_callbacks)

#test
# score = model.evaluate(test_generator,verbose=1)
# print('Test accuracy:', score[1])