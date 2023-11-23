import numpy as np
import pandas as pd

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from Configurations import *
from Logger import CustomLogger

import tensorflow as tf

logger = CustomLogger("INFO") # initializing logger

df = pd.read_csv(TEST_PATH)
df_valid = pd.read_csv(VALID_PATH)
df_test = pd.read_csv(TEST_PATH)

# Get the unique classes
unique_classes_train = df['class'].unique()
unique_classes_test = df_test['class'].unique()

# Determine the number of unique classes
num_classes = len(unique_classes_train)

logger.log("Number of unique classes for train datset: ", num_classes,level="INFO")
logger.log("Number of unique classes for test dataset: ", len(unique_classes_test),level="INFO")

logger.log("rescaling images",level="DEBUG")
datagen = ImageDataGenerator(rescale=1. / 255, )  # for rescaling and also a 20% validation split
datagen_valid = ImageDataGenerator(rescale=1. / 255)  # for rescaling and also a 20% validation split
datagen_test = ImageDataGenerator(rescale=1. / 255)  # for rescaling and also a 20% validation split


# Define the ImageDataGenerator with desired augmentations
logger.log("Image data generator with augmentation ",level="DEBUG")

datagen_augmented = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,  # Rotate the image by a random value between 0 and 40 degrees
    width_shift_range=0.2,  # Shift the image horizontally by a fraction of its width
    height_shift_range=0.2,  # Shift the image vertically by a fraction of its height
    shear_range=0.2,  # Apply shear transformations
    zoom_range=0.2,  # Zoom into the image by a random value up to 20%
    horizontal_flip=True,  # Allow horizontal flipping
    fill_mode='nearest'  # How to fill in missing pixels after a transformation
)
train_generator_augmented = datagen_augmented.flow_from_dataframe(
    dataframe=df,
    directory='../datasets/dataset2/train/',
    x_col="filename",
    y_col="class",
    batch_size=32,
    shuffle=True,
    class_mode="categorical",
    target_size=(150, 150)
)

"""
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='../datasets/dataset2/train/',
    x_col="filename",
    y_col="class",
    batch_size=32,
    shuffle=True,
    class_mode="categorical", # or "binary" if you have just 2 classes
    target_size=(150,150)
)
"""

validation_generator = datagen_valid.flow_from_dataframe(
    dataframe=df_valid,
    directory='../datasets/dataset2/valid/',
    x_col="filename",
    y_col="class",
    batch_size=32,
    shuffle=True,
    class_mode="categorical",
    target_size=(150, 150)
)

test_generator = datagen_test.flow_from_dataframe(
    dataframe=df_test,
    directory='../datasets/dataset2/test/',
    x_col="filename",
    y_col="class",
    batch_size=32,
    shuffle=False,
    class_mode="categorical",
    target_size=(150, 150)
)

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

dir_train_generator = train_datagen.flow_from_directory(
    directory='../datasets/dataset1/train/',
    target_size=(150, 150),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)


# Custom generator to combine both datasets
def combined_generator(gen1, gen2):
    while True:
        if np.random.choice([True, False]):  # Randomly choose which generator to yield from
            yield next(gen1)
        else:
            yield next(gen2)


train_gen_combined = combined_generator(train_generator_augmented, dir_train_generator)

""" Model simple CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax') # Use num_classes instead of hardcoding the number
])
"""
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
output_layer = Dense(num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(train_generator, validation_data=validation_generator, epochs=20)
# model.fit(train_generator_augmented, validation_data=validation_generator, epochs=150)

early_stop = EarlyStopping(monitor='val_loss', patience=10)  # patience is the number of epochs with no improvement
# model.fit(train_generator_augmented, validation_data=validation_generator, epochs=150, callbacks=[early_stop])
# model.fit(train_gen_combined, validation_data=validation_generator,
#          steps_per_epoch=(len(df) + dir_train_generator.samples) // BATCH_SIZE, 
#          epochs=EPOCHS)

history = model.fit(train_gen_combined, validation_data=validation_generator,
                    steps_per_epoch=(len(df) + dir_train_generator.samples) // BATCH_SIZE,
                    epochs=EPOCHS)

model.save('model_se_net.h5')
import pickle

# Save the history object to a file
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

eval_result = model.evaluate(test_generator)
logger.log("[test loss, test accuracy]:", eval_result,level="INFO")
