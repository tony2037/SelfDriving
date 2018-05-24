import cv2, glob


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


if __name__ == "__main__":
    resize_224x224()