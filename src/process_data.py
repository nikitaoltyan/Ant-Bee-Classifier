# Nikita Oltyan
from keras.preprocessing.image import ImageDataGenerator


def prepare_train_data(IMG_SHAPE=(180, 180), BATCH_SIZE=32,
                       VAL_SPLIT=0.2, RESCALE=1/255, ROTATION_RANGE=8,
                       WIDTH_SHIFT_RANGE=0.15, HEIGHT_SHIFT_RANGE=0.15,
                       ZOOM_RANGE=0.15, BRIGHTNESS_RANGE=(0.7, 1.3),
                       HORIZONTAL_FLIP=True):

    train_datagen = ImageDataGenerator(
        rescale=RESCALE,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        zoom_range=ZOOM_RANGE,
        brightness_range=BRIGHTNESS_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        validation_split=VAL_SPLIT
    )

    # Generators
    train_generator = train_datagen.flow_from_directory(
        '../data/raw/hymenoptera_data/train',
        target_size=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        '../data/raw/hymenoptera_data/train',
        target_size=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    data_shape = (IMG_SHAPE[0], IMG_SHAPE[1], 3)


    return train_generator, val_generator, data_shape


def prepare_test_data():
    return None, None
