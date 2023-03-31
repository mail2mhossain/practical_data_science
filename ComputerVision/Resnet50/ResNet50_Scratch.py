import tensorflow as tf
from tensorflow.keras import layers

def residual_block(x, filters, stride=1, downsample=None):
    residual = x
    identity = x

    # First convolution layer
    x = layers.Conv2D(filters=filters, kernel_size=(1,1), strides=(stride,stride), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second convolution layer
    x = layers.Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Third convolution layer
    x = layers.Conv2D(filters=4*filters, kernel_size=(1,1), strides=(1,1), padding='valid')(x)
    x = layers.BatchNormalization()(x)

    if downsample is not None:
        residual = downsample(identity)

    x += residual
    x = layers.Activation('relu')(x)
    return x

def resnet50(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution layer
    x = layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)

    # Residual blocks
    x = residual_block(x, filters=64, stride=1)
    x = residual_block(x, filters=64, stride=1)
    x = residual_block(x, filters=64, stride=1)

    x = residual_block(x, filters=128, stride=2, downsample=layers.Conv2D(filters=128, kernel_size=(1,1), strides=(2,2), padding='valid'))
    x = residual_block(x, filters=128, stride=1)
    x = residual_block(x, filters=128, stride=1)
    x = residual_block(x, filters=128, stride=1)

    x = residual_block(x, filters=256, stride=2, downsample=layers.Conv2D(filters=256, kernel_size=(1,1), strides=(2,2), padding='valid'))
    x = residual_block(x, filters=256, stride=1)
    x = residual_block(x, filters=256, stride=1)
    x = residual_block(x, filters=256, stride=1)
    x = residual_block(x, filters=256, stride=1)
    x = residual_block(x, filters=256, stride=1)

    x = residual_block(x, filters=512, stride=2, downsample=layers.Conv2D(filters=512, kernel_size=(1,1), strides=(2,2), padding='valid'))
    x = residual_block(x, filters=512, stride=1)
    x = residual_block(x, filters=512, stride=1)

    # Final fully connected layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name='resnet50')
    return model
