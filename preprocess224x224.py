import cv2, glob
import numpy as np


def resize_224x224():
    x_path = "./dataset/leftImg8bit/train/bremen/"
    y_path = "./dataset/gtFine/train/bremen/"

    x_save_path = "./dataset/dataset224x224/x/"
    y_save_path = "./dataset/dataset224x224/y/"

    counter = 0

    x_list = glob.glob(x_path + "*.png")
    y_list = []
    for i in x_list:
        y_list.append(y_path+ i[35:-16]+ "_gtFine_color.png")

    trainset = [(a,b) for a,b in zip(x_list, y_list)]

    for i in trainset:
        pic_x = cv2.imread(i[0])
        pic_x = cv2.resize(pic_x, (224, 224), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(x_save_path+ str(counter)+ ".png", pic_x)
        pic_y = cv2.imread(i[1])
        pic_y = cv2.resize(pic_y, (224, 224), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(y_save_path+ str(counter)+ ".png", pic_y)
        counter = counter + 1

    #pic = cv2.imread('./Elegent_Girl.jpg')
    #pic = cv2.resize(pic, (224, 224), interpolation=cv2.INTER_CUBIC)

def BGR_to_one_hot(BGR):
    """
    label = {
        "car": [142, 0, 0],
        "road": [128, 64, 128],
        "sky": [180, 130, 70],
        "parking": [160, 170, 250],
        "else": []
    }
    BGR
    """
    if((BGR == [142, 0, 0]).all()):
        return [1, 0, 0, 0, 0]
    elif((BGR == [128, 64, 128]).all()):
        return [0, 1, 0, 0, 0]
    elif((BGR == [180, 130, 0]).all()):
        return [0, 0, 1, 0, 0]
    elif((BGR == [160, 170, 250]).all()):
        return [0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 1]

def y_to_npy():
    """
    label = {
        "car": [142, 0, 0],
        "road": [128, 64, 128],
        "sky": [180, 130, 70],
        "parking": [160, 170, 250],
        "else": []
    }
    BGR
    """
    y_path = "./dataset/dataset224x224/y/"
<<<<<<< HEAD
    image_number = 78
    y_list = []
    for i in range(0, image_number):
=======
    image_number = 77
    y_list = []
    for i in range(0, image_number+1):
>>>>>>> 4d881198dd405bd894beda002b06ec6c35efb7a2
        y_list.append(y_path + str(i) + ".png")
    
    for i in y_list:
        print(i[27:-4])
        pic = cv2.imread(i)
        encoded = np.zeros(( len(pic), len(pic[0]), 5))
        for k in range(0, len(pic)):
            for j in range(0, len(pic[k])):
                encoded[k][j] = BGR_to_one_hot(pic[k][j])
        assert encoded.shape == (224, 224, 5)
        np.save(y_path + i[27:-4] + ".npy", encoded)

<<<<<<< HEAD
def load_data(total_number=78):
    x_path = "./dataset/dataset224x224/x/"
    y_path = "./dataset/dataset224x224/y/"

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    for i in range(0, total_number):
        x = cv2.imread(x_path + str(i) + ".png")    
        x_train.append(x)

        y = np.load(y_path + str(i) + ".npy")
        y_train.append(y)
    assert x_train[0].shape == (224,224,3)
    assert y_train[0].shape == (224,224,5)

    return (x_train, y_train), (x_test, y_test)


=======
def load_data(total_number=77):
    x_path = "./dataset/dataset224x224/x/"
    y_path = "./dataset/dataset224x224/y/"
    for i in range(0, total_number+1):
        
>>>>>>> 4d881198dd405bd894beda002b06ec6c35efb7a2

if __name__ == "__main__":
    #resize_224x224()
    #y_to_npy()
<<<<<<< HEAD
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    (x_train, y_train), (x_test, y_test) = load_data(78)
=======
    load_data(77)
>>>>>>> 4d881198dd405bd894beda002b06ec6c35efb7a2
