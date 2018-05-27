from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, UpSampling2D, Conv2DTranspose
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import keras
from preprocess224x224 import load_data
import numpy as np



def train(x_train, y_train, x_test, y_test):
    print(x_train[77].shape)
    print(y_train[77].shape)
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
    model.add(Conv2D(5, (1, 1),strides=(1, 1), padding='same', activation='softmax'))
    model.summary()

    model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

    # keras callback function
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
    checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
    
    train_history = model.fit(x=x_train,  
                          y=y_train, validation_split=0.2,  
                          epochs=100, batch_size=1, verbose=2, callbacks=[reduce_lr, checkpointer])
    model.save('Deconvolution.h5')  # creates a HDF5 file 'Deconvolution.h5'
    #Deconvolution2D(3, 3, 3, output_shape=(None, 3, 14, 14),border_mode='valid',input_shape=(3, 12, 12))
    #model.add(UpSampling2D(size=(2, 2),input_shape=image_size))

if __name__=="__main__":
    
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    (x_train, y_train), (x_test, y_test) = load_data(916)
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    train(x_train, y_train, x_test, y_test)
