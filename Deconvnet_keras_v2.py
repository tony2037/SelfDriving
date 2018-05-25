from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, UpSampling2D, Conv2DTranspose
import keras




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
    #model.add(Flatten())
    #model.add(Dense(4096))
    #model.add(Dense(4096))
    model.add(Conv2D(4096, (7, 7),strides=(1, 1), padding='valid'))
    model.add(Conv2D(4096, (1, 1),strides=(1, 1), padding='same'))
    model.add(Conv2DTranspose(512, (7,7), padding='valid'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(512, (3,3), padding='same'))
    model.add(Conv2DTranspose(512, (3,3), padding='same'))
    model.add(Conv2DTranspose(512, (3,3), padding='same'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(512, (3,3), padding='same'))
    model.add(Conv2DTranspose(512, (3,3), padding='same'))
    model.add(Conv2DTranspose(256, (3,3), padding='same'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(256, (3,3), padding='same'))
    model.add(Conv2DTranspose(256, (3,3), padding='same'))
    model.add(Conv2DTranspose(128, (3,3), padding='same'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(128, (3,3), padding='same'))
    model.add(Conv2DTranspose(64, (3,3), padding='same'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(64, (3,3), padding='same'))
    model.add(Conv2DTranspose(64, (3,3), padding='same'))
    model.add(Conv2D(21, (1, 1),strides=(1, 1), padding='same'))
    model.summary()

    model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

    #Deconvolution2D(3, 3, 3, output_shape=(None, 3, 14, 14),border_mode='valid',input_shape=(3, 12, 12))
    #model.add(UpSampling2D(size=(2, 2),input_shape=image_size))