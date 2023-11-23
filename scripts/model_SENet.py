#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

from Configurations import *
from Logger import CustomLogger
from potato import PotatoLeavesClassifier

import pickle

logger = CustomLogger("INFO")  # initializing logger


""" Model SENet"""
def se_block(input_tensor, ratio=16):
    """The Squeeze-and-Excitation block.
    Args:
    - input_tensor: input tensor to the block
    - ratio: reduction ratio for the number of filters. Default is 16.

    Returns:
    - Output tensor after being scaled.
    """
    init = input_tensor
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    # Squeeze
    se = tf.keras.layers.GlobalAveragePooling2D()(init)
    se = tf.keras.layers.Reshape(se_shape)(se)

    # Excitation
    se = tf.keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    # Scale the input
    return tf.keras.layers.multiply([init, se])

if __name__=="__main__":

    model_SENet = PotatoLeavesClassifier()
    model_SENet.load_dataset()
    # Now, let's build the model using the Functional API:
    input_layer = tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D(2, 2)(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = se_block(x)  # Add the SE block
    x = MaxPooling2D(2, 2)(x)
    
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = se_block(x)  # Add the SE block
    x = MaxPooling2D(2, 2)(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output_layer = Dense(int(model_SENet.num_classes), activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # model.fit(train_generator, validation_data=validation_generator, epochs=20)
    # model.fit(train_generator_augmented, validation_data=validation_generator, epochs=150)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10)  # patience is the number of epochs with no improvement
    # model.fit(train_generator_augmented, validation_data=validation_generator, epochs=150, callbacks=[early_stop])
    # model.fit(train_gen_combined, validation_data=validation_generator,
    #          steps_per_epoch=(len(df) + dir_train_generator.samples) // BATCH_SIZE,
    #          epochs=EPOCHS)
    
    history = model.fit(model_SENet.train_gen_combined, validation_data=model_SENet.validation_generator,
                        steps_per_epoch=(len(model_SENet.df_train_1) + model_SENet.dir_train_generator.samples) // BATCH_SIZE,
                        epochs=EPOCHS)

    model.save('model_se_net.h5')

    # Save the history object to a file
    with open('history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    eval_result = model.evaluate(model_SENet.test_generator)
    logger.log("[test loss, test accuracy]:", eval_result, level="INFO")
