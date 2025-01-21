
import tensorflow as tf
from model_cifar10 import build_cifar10_model
from utils import load_cifar10

from tf.keras.optimizers import Adam
from tf.keras.preprocessing.image import ImageDataGenerator
from tf.keras.callbacks import ReduceLROnPlateau

# Load data
(x_train, y_train), (x_test, y_test) = load_cifar10()

# Convert labels to categorical (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Load model
model = build_cifar10_model()

# Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)


# Train model
history = model.fit(datagen.flow(x_train, y_cat_train, batch_size=32),
                    validation_data=(x_test, y_cat_test),
                    epochs=50, callbacks=[lr_scheduler])

# Save trained model
model.save("model_)cifar10_imp.h5")
model.save("model_cifar10_imp.keras")