# For Moving Data
import os
import os.path
import shutil
import random

# For Preprocess input Image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# input image size
IMAGE_SIZE = (224, 224)

DATASET_PARENT_FOLDER = 'Sign-Language-Digits-Dataset-master'
DATASET_FOLDER = f'{DATASET_PARENT_FOLDER}/Dataset'
TRAINING_FOLDER = 'sign/train'
VALIDATION_FOLDER = 'sign/valid'

def pre_process_dataset():
    """
    Seprate the input dataset into training & validation dataset.
    """
    # Create The Parent Directory if not exists
    if not os.path.isdir('sign'):
        os.mkdir('sign')

    # Create The Training Directory if not exists
    if not os.path.isdir(TRAINING_FOLDER):
        os.mkdir(TRAINING_FOLDER)

    # Create The Validation Directory if not exists
    if not os.path.isdir(VALIDATION_FOLDER):
        os.mkdir(VALIDATION_FOLDER)

    for i in range(0, 10):
        # Input Dataset folder for particular digit
        folder = f'{DATASET_FOLDER}/{i}'

        # Check if dataset for particular number exists. FIXME: We can also check if data is already copied
        if not os.path.isdir(folder):
            print('Dataset Directory not found!')
            continue

        # List all The files on the dataset directory
        files = os.listdir(folder)

        # Get 20 random values for validation dataset
        # 20 ~ 10% of the original dataset
        valid = random.sample(files, 20)

        # Destination Folder for validation dataset
        dst_folder = f'{VALIDATION_FOLDER}/{i}'

        # if the destination not exist
        if not os.path.isdir(dst_folder):
            # Create Destination Folder
            os.mkdir(dst_folder)

            # All the file chosen for validation dataset move them to destination
            for file in valid:
                shutil.move(f'{folder}/{file}', f'{dst_folder}/{file}')

            # Move the remaing dataset to training folder
            shutil.move(folder, f'{TRAINING_FOLDER}/{i}')


def get_image_gen():
    """
    preproces input images from the directory.
    """
    # 2.24 => 224 / 100 (100 is image size)

    train_datagen = ImageDataGenerator(
        rescale=2.24,
        preprocessing_function=preprocess_input
    )

    valid_datagen = ImageDataGenerator(
        rescale=2.24,
        preprocessing_function=preprocess_input
    )
    # 10 batches 0 - 9

    train_set = train_datagen.flow_from_directory(
        directory='sign/train',
        target_size=IMAGE_SIZE,
        batch_size=10,
        class_mode='categorical'
    )

    valid_set = valid_datagen.flow_from_directory(
        directory='sign/valid',
        target_size=IMAGE_SIZE,
        batch_size=10,
        class_mode='categorical'
    )

    return train_set, valid_set
