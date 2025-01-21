# CNN Model
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, 
                                     Dropout, GlobalAveragePooling2D, Add, Input, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

def build_cifar10_model():
    """Build and return model"""
    # Define Residual Block
    def residual_block(x, filters):
        """Creates a Residual Block with a 1x1 Conv for matching dimensions."""
        shortcut = x  # Save input for shortcut connection
        
        x = Conv2D(filters, (3,3), activation='swish', padding='same')(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(filters, (3,3), activation='swish', padding='same')(x)
        x = BatchNormalization()(x)

        # **Fix:** Apply 1x1 Conv to match shortcut dimensions if needed
        if shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)
        
        x = Add()([x, shortcut])  # Add skip connection
        return x

    # Define Model Architecture
    input_layer = Input(shape=(32, 32, 3))  # Single input (can be expanded later)

    x = Conv2D(32, (3,3), activation='swish', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = residual_block(x, 64)
    x = MaxPooling2D((2,2))(x)

    x = residual_block(x, 128)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)  # Flatten to preserve more feature information
    x = Dense(256, activation='swish')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001),
                   metrics=['accuracy'])
    return model