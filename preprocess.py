import numpy as np
import cv2
from os import walk

#labels = {
#[0,0,0]:["car", [1, 0, 0, 0, 0]],
#[128, 64,128]:["road", [0, 1, 0, 0, 0]],
#[180,130,70]:["sky", [0, 0, 1, 0, 0]],
#[160,170,250]:["parking", [0, 0, 0, 1, 0]]
#}
#[]:["no-road", [0, 0, 0, 0, 1]]


#cv2.imshow("Fist image", img)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Various methods of cv resize
methods = [
    ("cv2.INTER_NEAREST", cv2.INTER_NEAREST),
    ("cv2.INTER_LINEAR", cv2.INTER_LINEAR),
    ("cv2.INTER_AREA", cv2.INTER_AREA),
    ("cv2.INTER_CUBIC", cv2.INTER_CUBIC),
    ("cv2.INTER_LANCZOS4", cv2.INTER_LANCZOS4)]

def classify(rgb_arr):
    if((rgb_arr == [0,0,0]).all()):
        return  [1, 0, 0, 0, 0]
    elif((rgb_arr == [128, 64,128]).all()):
        return [0, 1, 0, 0, 0]
    elif((rgb_arr == [180,130,70]).all()):
        return [0, 0, 1, 0, 0]
    elif((rgb_arr == [160,170,250]).all()):
        return [0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 1]

#print(classify(img[110][120]))

def resize(image, width=None, height=None, inter=cv2.INTER_LINEAR):
    # 初始化, 取得shape
    dim = None
    (h, w) = image.shape[:2]

    # width == 0 && height == 0 ,then return image
    if width is None and height is None:
        return image

    # 寬度是0
    if width is None:
        # 根據高度縮放比例
        r = height / float(h)
        dim = (int(w * r), height)

    # 高度是0
    else:
        # 根據寬度縮放比例
        r = width / float(w)
        dim = (width, int(h * r))

    # 縮放圖像
    resized = cv2.resize(image, dim, interpolation=inter)

    # 返回縮放後的image
    return resized

def demo():
    #img = cv2.imread("./gtFine/train/darmstadt/darmstadt_000000_000019_gtFine_color.png")

    #print(img.shape)

    #print(img[110][120])

    try_resize = cv2.imread("./dataset/leftImg8bit/train/bremen/bremen_000002_000019_leftImg8bit.png")
    cv2.imshow("原圖", try_resize)

    rotate = resize(try_resize, width= int(try_resize.shape[1] /4))
    cv2.imshow("翻轉圖", rotate)
    cv2.waitKey(0)

    try_y_resize = cv2.imread("./dataset/gtFine/train/bremen/bremen_000000_000019_gtFine_color.png")
    cv2.imshow("答案", try_y_resize)

    y_rotate = resize(try_y_resize, width= int(try_y_resize.shape[1] /4))
    cv2.imshow("翻轉答案", y_rotate)
    cv2.waitKey(0)

def refine():
    list_x = []
    dirpath_x = ''
    save_path_x = "./dataset/preprocess_image/x/"

    list_y = []
    dirpath_y = ''
    save_path_y = './dataset/preprocess_image/y/'


    for (dirpath, dirnames, filenames) in walk("./dataset/leftImg8bit/train/bremen"):
        #print(dirpath, dirnames, filenames)
        dirpath_x = dirpath
        list_x.extend(filenames)
        break
    
    for x in list_x:
        print(dirpath_x + "/" + x)
        ToBeResize = cv2.imread((dirpath_x + "/" + x))
        print(ToBeResize.shape)
        BeResize = resize(ToBeResize, width=int(ToBeResize.shape[1] /4))
        cv2.imwrite(save_path_x + x, BeResize)

    for (dirpath, dirnames, filenames) in walk("./dataset/gtFine/train/bremen"):
        #print(dirpath, dirnames, filenames)
        dirpath_y = dirpath
        list_y.extend(filenames)
        break

    for y in list_x:
        _y = y[0:-15] + "gtFine_color.png"
        print(dirpath_y + "/" + _y)
        ToBeResize = cv2.imread((dirpath_y + "/" + _y))
        print(ToBeResize.shape)
        BeResize = resize(ToBeResize, width=int(ToBeResize.shape[1] /4))
        cv2.imwrite(save_path_y + _y, BeResize)

def transform_to_npy():
    """
    transform images to npy:
    dealing_path : the path of the directory dealing .(The "/" at the last is nessassary)
    
    output:
    save as a .npy file 
    """
    dealing_path = './dataset/preprocess_image/y/'
    dealing_list = []
    output = []
    for (dirpath, dirnames, filenames) in walk(dealing_path):
        dealing_list.extend(filenames)

    for dealing_y in dealing_list:
        img = cv2.imread(dealing_path + dealing_y)
        result_img = np.zeros((img.shape[1], img.shape[0],5), dtype=float, order='C')
        
        print(img.shape)
        for h in range(0,img.shape[1]): #(h=0;h<img.shape[1];h++):
            for w in range(0,img.shape[0]): #(w=0;w<img.shape[0];w++):
                result_img[h][w] = classify(img[w][h])
        
        print(result_img.shape)
        output.append(result_img)

        if( np.array(result_img).shape == (512,256,5)):
            print("dimmension check . Saving a" + dealing_y + ".npy file")
            np.save("./dataset/preprocess_image/ys.npy/" + dealing_y + ".npy", result_img)
        else:
            print("dimmension error")
            print(np.array(result_img).shape)
        
    if( np.array(output).shape == (78,512,256,5)):
        print("dimmension check")
        np.save("./dataset/preprocess_image/y/y.npy", output)
    else:
        print("dimmension error")
        print(np.array(output).shape)

def transform_to_npy_v2():
    """
    transform images to npy:
    dealing_path : the path of the directory dealing .(The "/" at the last is nessassary)
    
    output:
    save as a .npy file 
    """
    dealing_path = './dataset/preprocess_image/y/'
    dealing_list = []
    output = []
    for (dirpath, dirnames, filenames) in walk(dealing_path):
        dealing_list.extend(filenames)

    for dealing_y in dealing_list:
        img = cv2.imread(dealing_path + dealing_y)
        assert img.shape == (256,512,3)
        result_img = np.zeros((img.shape[0], img.shape[1],5), dtype=float, order='C')
        
        print(img.shape)
        for h in range(0,img.shape[0]): #(h=0;h<img.shape[0];h++):
            for w in range(0,img.shape[1]): #(w=0;w<img.shape[1];w++):
                result_img[h][w] = classify(img[h][w])
        
        print(result_img.shape)
        assert result_img.shape == (256,512,5)
        output.append(result_img)

        if( np.array(result_img).shape == (256,512,5)):
            print("dimmension check . Saving a" + dealing_y + ".npy file")
            np.save("./dataset/preprocess_image/ys.npy/" + dealing_y + ".npy", result_img)
        else:
            print("dimmension error")
            print(np.array(result_img).shape)
        
    if( np.array(output).shape == (78,512,256,5)):
        print("dimmension check")
        np.save("./dataset/preprocess_image/y/y.npy", output)
    else:
        print("dimmension error")
        print(np.array(output).shape)

if __name__ == '__main__':
    transform_to_npy_v2()