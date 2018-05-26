import cv2, glob
import numpy as np


def resize_224x224():
    x_path = "./dataset/leftImg8bit/train/zurich/"
    y_path = "./dataset/gtFine/train/zurich/"

    x_save_path = "./dataset/dataset224x224/x_/"
    y_save_path = "./dataset/dataset224x224/y_/"

    log_path = "./dataset/dataset224x224/x/log"
    log = open(log_path)
    counter = log.read()
    counter = int(counter)
    log.close()

    x_list = glob.glob(x_path + "*.png")
    y_list = []

    log = open(log_path, "w")
    log.write(str(counter+ len(x_list)))

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
    log_path = "./dataset/dataset224x224/y/log"
    log = open(log_path)
    start_number = log.read()
    start_number = int(start_number)
    log.close()
    image_number = int(open("./dataset/dataset224x224/x/log").read()) - start_number
    log = open(log_path, "w")
    log.write(str(start_number+image_number))
    y_list = []
    for i in range(start_number, start_number+image_number):
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

def load_data(total_number=916):
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



if __name__ == "__main__":
    #resize_224x224()
    y_to_npy()
    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    (x_train, y_train), (x_test, y_test) = load_data(78)
    """
    """
    x_list = glob.glob("./dataset/leftImg8bit/train/zurich/" + "*.png")
    for i in x_list:
        print(i[35:-16])
    """