import pandas as pd

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from Configurations import *
from Logger import CustomLogger
from helper_functions import *

import tensorflow as tf

logger = CustomLogger("INFO")  # initializing logger


class PotatoLeavesClassifier:
    def __init__(self):
        self.train_generator_augmented_1 = None
        self.validation_generator = None
        self.test_generator = None
        self.train_gen_combined = None
        self.num_classes = None
        self.df_train_1 = None
        self.df_valid = None
        self.df_test = None

    def load_dataset(self):
        self.df_train_1 = pd.read_csv(TRAIN_PATH_1)
        self.df_valid = pd.read_csv(VALID_PATH)
        self.df_test = pd.read_csv(TEST_PATH)

        # Get the unique classes
        unique_classes_train = self.df_train_1['class'].unique()
        unique_classes_test = self.df_test['class'].unique()

        # Determine the number of unique classes
        self.num_classes = len(unique_classes_train)

        logger.log("Number of unique classes for train datset: ", self.num_classes, level="INFO")
        logger.log("Number of unique classes for test dataset: ", len(unique_classes_test), level="INFO")

        logger.log("rescaling images", level="DEBUG")
        datagen = ImageDataGenerator(rescale=1. / 255, )  # for rescaling and also a 20% validation split
        datagen_valid = ImageDataGenerator(rescale=1. / 255)  # for rescaling and also a 20% validation split
        datagen_test = ImageDataGenerator(rescale=1. / 255)  # for rescaling and also a 20% validation split

        # Define the ImageDataGenerator with desired augmentations
        logger.log("Image data generator with augmentation ", level="DEBUG")

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

        # NOTE: only train datasets are augmented
        self.train_generator_augmented_1 = datagen_augmented.flow_from_dataframe(
            dataframe=self.df_train_1,
            directory='../datasets/dataset2/train/',
            x_col="filename",
            y_col="class",
            batch_size=32,
            shuffle=True,
            class_mode="categorical",
            target_size=(150, 150)
        )

        # NOTE: the validation datasets are not augmented
        self.validation_generator = datagen_valid.flow_from_dataframe(
            dataframe=self.df_valid,
            directory='../datasets/dataset2/valid/',
            x_col="filename",
            y_col="class",
            batch_size=32,
            shuffle=True,
            class_mode="categorical",
            target_size=(150, 150)
        )

        # NOTE: the test datasets are not augmented
        self.test_generator = datagen_test.flow_from_dataframe(
            dataframe=self.df_test,
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

        self.dir_train_generator = train_datagen.flow_from_directory(
            directory='../datasets/dataset1/train/',
            target_size=(150, 150),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )

        # Custom generator to combine both datasets
        def combined_generator( gen1, gen2):
            while True:
                if np.random.choice([True, False]):  # Randomly choose which generator to yield from
                    yield next(gen1)
                else:
                    yield next(gen2)

        self.train_gen_combined = combined_generator(self.dir_train_generator, self.train_generator_augmented_1)



