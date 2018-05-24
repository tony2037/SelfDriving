from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten





if __name__ == "__main__":
    image_size = (224,224,3)
    model = Sequential()
    model.add(Conv2D(64, (3, 3),strides=(1, 1), padding='same', input_shape=image_size))
    model.add(Conv2D(64, (3, 3),strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3),strides=(1, 1), padding='same'))
    model.add(Conv2D(128, (3, 3),strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(256, (3, 3),strides=(1, 1), padding='same'))
    model.add(Conv2D(256, (3, 3),strides=(1, 1), padding='same'))
    model.add(Conv2D(256, (3, 3),strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(512, (3, 3),strides=(1, 1), padding='same'))
    model.add(Conv2D(512, (3, 3),strides=(1, 1), padding='same'))
    model.add(Conv2D(512, (3, 3),strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(512, (3, 3),strides=(1, 1), padding='same'))
    model.add(Conv2D(512, (3, 3),strides=(1, 1), padding='same'))
    model.add(Conv2D(512, (3, 3),strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Dense(4096))
    model.summary()