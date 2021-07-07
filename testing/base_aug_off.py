# Import packages and set numpy random seed
import tensorflow as tf
import numpy as np
np.random.seed(5) 
import cv2
import keras
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Reshape, Permute, Multiply, Input, Activation
from keras.models import Sequential, Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    '../../ddd_images_test/all',
    target_size=(128, 128),
    batch_size=32,
    shuffle=True,
    class_mode='binary')



model = Sequential(
[
    keras.Input((128,128,3)),
    Conv2D(32, 5, activation='relu', padding = 'same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(32, 5, activation='relu', padding = 'same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.5),

    Conv2D(16, 5, activation='relu', padding = 'same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(8, 5, activation='relu', padding = 'same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.5),

    Conv2D(4, 5, activation='relu', padding = 'same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(1, activation='sigmoid')
]
)
optimizer = keras.optimizers.Adam()
val_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True) #from_logits=True means output probabilities are not normalized
acc_metric = keras.metrics.BinaryAccuracy()
val_acc_metric = keras.metrics.BinaryAccuracy()

model.summary()
model.load_weights('models/data_aug_off_model.h5')

@tf.function
def test_step(x, y):
    val_preds = model(x, training=False)
    loss = val_loss_fn(y, val_preds)
    
    # Update val metrics
    val_acc_metric.update_state(y, val_preds)
    return val_preds, loss

valid_writer = tf.summary.create_file_writer('logs/base_aug_off_test/test', max_queue = 10)

total_valid_files = 0

for val_batch_idx in range(test_generator.samples//32):
    recent_test_batch = test_generator.next()
    x_test_batch = recent_test_batch[0]
    y_test_batch = recent_test_batch[1]

    val_y_preds,val_loss = test_step(x_test_batch,y_test_batch)
    val_y_pred = np.reshape(tf.get_static_value(val_y_preds),(1,32))[0]
    test_accuracy = val_acc_metric.result()
    if val_batch_idx%32 == 0 and val_batch_idx!=0:
        with valid_writer.as_default():
            tf.summary.scalar("test_accuracy", test_accuracy, step = total_valid_files)
        print("Test Accuracy for batch {} is: ".format(val_batch_idx), test_accuracy)
        total_valid_files += 32