import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


import pandas as pd

IMG_SIZE = (150,150)
BATCH_SIZE = 32

df = pd.read_csv('../datasets/dataset2/train/_annotations.csv')
df_valid = pd.read_csv('../datasets/dataset2/valid/_annotations.csv')
df_test= pd.read_csv('../datasets/dataset2/test/_annotations.csv')
# Get the unique classes
unique_classes_train = df['class'].unique()
unique_classes_test = df_test['class'].unique()
# Determine the number of unique classes
num_classes = len(unique_classes_train)

print("Number of unique classes:\n", num_classes)
print("Number of unique classes test:\n", len(unique_classes_test))


datagen = ImageDataGenerator(rescale=1./255, ) # for rescaling and also a 20% validation split
datagen_valid = ImageDataGenerator(rescale=1./255) # for rescaling and also a 20% validation split
datagen_test = ImageDataGenerator(rescale=1./255) # for rescaling and also a 20% validation split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the ImageDataGenerator with desired augmentations
datagen_augmented = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,       # Rotate the image by a random value between 0 and 40 degrees
    width_shift_range=0.2,   # Shift the image horizontally by a fraction of its width
    height_shift_range=0.2,  # Shift the image vertically by a fraction of its height
    shear_range=0.2,         # Apply shear transformations
    zoom_range=0.2,          # Zoom into the image by a random value up to 20%
    horizontal_flip=True,    # Allow horizontal flipping
    fill_mode='nearest'      # How to fill in missing pixels after a transformation
)
train_generator_augmented = datagen_augmented.flow_from_dataframe(
    dataframe=df,
    directory='../datasets/dataset2/train/',
    x_col="filename",
    y_col="class",
    batch_size=32,
    shuffle=True,
    class_mode="categorical",
    target_size=(150,150)
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
    target_size=(150,150)
)

test_generator = datagen_test.flow_from_dataframe(
    dataframe=df_test,
    directory='../datasets/dataset2/test/',
    x_col="filename",
    y_col="class",
    batch_size=32,
    shuffle=False,
    class_mode="categorical",
    target_size=(150,150)
)

train_datagen = ImageDataGenerator(rescale=1./255,
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

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Define constants
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Define the number of classes (based on your previous code snippet)
num_classes = len(unique_classes_train)

# Load the ResNet50 model without the top layer
base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Freeze the layers of the ResNet50 model to retain their pretrained weights
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling to reduce spatial dimensions
x = Dense(512, activation='relu')(x)  # Dense layer for further learning from the dataset
predictions = Dense(num_classes, activation='softmax')(x)  # Final softmax layer for classification

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen_combined, validation_data=validation_generator, 
          steps_per_epoch=(len(df) + dir_train_generator.samples) // BATCH_SIZE, 
          epochs=150)
 
model.save('model_ress_net101.h5')


eval_result = model.evaluate(test_generator)
print("[test loss, test accuracy]:", eval_result)

