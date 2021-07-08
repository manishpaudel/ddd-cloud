import keras
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Reshape, Permute, Multiply, Input, Activation
from keras.models import Sequential, Model
import tensorflow.keras.backend as K

def SqueezeExcite(x, ratio=16, name=''):
    nb_chan = K.int_shape(x)[-1]

    y = GlobalAveragePooling2D(name='SE_avg_{}'.format(name))(x)
    y = Dense(nb_chan // ratio, activation='relu', name='se_dense1_{}'.format(name))(y)
    y = Dense(nb_chan, activation='sigmoid', name='dense2_{}'.format(name))(y)

    y = Multiply(name='se_mul_{}'.format(name))([x, y])
    return y

def SE_Conv2D_block(x, filters, kernel_size, name):
    y = Conv2D(filters, kernel_size, padding='same', name='conv_{}'.format(name))(x)
    #squeeze excite in this
    y = SqueezeExcite(y, ratio=16, name=name)
    #other relu and maxpools
    y = BatchNormalization(name='BatchNorm_{}'.format(name))(y)
    y = Activation('relu', name='Activation_{}'.format(name))(y)
    y = MaxPool2D(pool_size=(2, 2), padding='same', name='{}_pool'.format(name))(y)
    y = Dropout(0.25)(y)
    return y


def SE_Dense_block(x, size, name, bn=True):
    y = Dense(size, name='dense_{}'.format(name))(x)
    if bn:
        y = BatchNormalization(name='BatchNorm_{}'.format(name))(y)
    y = Activation('relu', name='Activation_{}'.format(name))(y)
    return y

def SE_Model(nb_class, input_shape, include_top=True, weights = False):
    """SE net without the splitted stream."""

    img_input = Input(shape=input_shape)

    x = SE_Conv2D_block(img_input, 32, (3, 3), name='block1')
    

    params = [
        (64, (3, 3)),
        (128, (3, 3)),
        (256, (3, 3))
    ]

    for i, (filters, kernel_size) in enumerate(params, start=2):
        x = SE_Conv2D_block(x, filters, kernel_size, name='block{}'.format(i))

    if include_top:
        x = Flatten(name='flatten')(x)
        x = Dense(nb_class, activation='sigmoid')(x)

    model = Model(inputs=img_input, outputs=x)

    if weights:
        print('Loading')
        model.load_weights(weights)
        print("Weight Loaded")

    return model