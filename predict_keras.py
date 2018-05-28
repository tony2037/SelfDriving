from keras.models import Model, Sequential, load_model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, UpSampling2D, Conv2DTranspose
import cv2
import os
import numpy as np
import glob

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

def load_trained_model_with_FullModel(Model_json_path="./model/model.json", Weights_h5_path="./model/model.h5", test_x_path="./dataset/dataset224x224/test_x/test.png"):
    # test_x shape
    test_x = np.zeros((1,224,224,3))

    # load json and create model
    json_file = open('./model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./model/model.h5")
    print("Loaded model from disk")

    return model

def predict(model, test_x_path="./dataset/dataset224x224/test_x/", predict_save_path="./dataset/dataset224x224/test_y/"):
    test_x_list = glob(test_x_path+"*.png")
    test_x = np.zeros((len(test_x_list, 224, 224, 3)))
    for i in range(0, len(test_x_list)):
        test_x[i] = cv2.imread(test_x_list[i])
    predict = model.predict(test_x, verbose=1)


def load_trained_model_with_weight(weights_path="./model/weights.hdf5", test_x_path="./dataset/dataset224x224/test_x/test.png"):
   model = create_model()
   model.load_weights(weights_path)
   test_x = cv2.imread(test_x_path)
   predict = model.predict(test_x, verbose=1)
   print(type(predict))



if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    load_trained_model_with_FullModel()
