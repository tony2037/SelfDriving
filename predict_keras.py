from keras.models import Model, Sequential, load_model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, UpSampling2D, Conv2DTranspose
import cv2
import os
import numpy as np

def create_model():
    image_size = (224,224,3)
    model = Sequential()
    model.add(Conv2D(64, (3, 3),strides=(1, 1), padding='same', input_shape=image_size, activation='relu'))
    model.add(Conv2D(64, (3, 3),strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3),strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3),strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(256, (3, 3),strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3),strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3),strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(512, (3, 3),strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3),strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3),strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(512, (3, 3),strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3),strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3),strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    #model.add(Flatten())
    #model.add(Dense(4096))
    #model.add(Dense(4096))
    model.add(Conv2D(4096, (7, 7),strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(4096, (1, 1),strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2DTranspose(512, (7,7), padding='valid', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(512, (3,3), padding='same', activation='relu'))
    model.add(Conv2DTranspose(512, (3,3), padding='same', activation='relu'))
    model.add(Conv2DTranspose(512, (3,3), padding='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(512, (3,3), padding='same', activation='relu'))
    model.add(Conv2DTranspose(512, (3,3), padding='same', activation='relu'))
    model.add(Conv2DTranspose(256, (3,3), padding='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(256, (3,3), padding='same', activation='relu'))
    model.add(Conv2DTranspose(256, (3,3), padding='same', activation='relu'))
    model.add(Conv2DTranspose(128, (3,3), padding='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(128, (3,3), padding='same', activation='relu'))
    model.add(Conv2DTranspose(64, (3,3), padding='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(64, (3,3), padding='same', activation='relu'))
    model.add(Conv2DTranspose(64, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(5, (1, 1),strides=(1, 1), padding='same', activation='softmax'))
    model.summary()
    return model

<<<<<<< HEAD
def load_trained_model_with_FullModel(Model_path="Deconvolution.h5", test_x_path="./dataset/dataset224x224/test_x/test.png"):
    model = load_model(Model_path)
=======
def load_trained_model_with_FullModel(Model_json_path="./model/model.json", Weights_h5_path="./model/model.h5", test_x_path="./dataset/dataset224x224/test_x/test.png"):
    # load json and create model
    json_file = open('./model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./model/model.h5")
    print("Loaded model from disk")
>>>>>>> 396b236440faa0e4b4c63312d6be926eda67ef46
    test_x = cv2.imread(test_x_path)
    predict = model.predict([test_x], verbose=1)
    print(predict.shape)

def load_trained_model_with_weight(weights_path="./model/weights.hdf5", test_x_path="./dataset/dataset224x224/test_x/test.png"):
   model = create_model()
   model.load_weights(weights_path)
   test_x = cv2.imread(test_x_path)
   predict = model.predict(test_x, verbose=1)
   print(type(predict))



if __name__ == "__main__":
<<<<<<< HEAD
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    load_trained_model_with_FullModel()
=======
    load_trained_model_with_FullModel()
>>>>>>> 7a15ab499c8eb2bd22c4d05edd25cb1204bd0172
