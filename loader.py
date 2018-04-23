import glob
import random

class loader(self, x_path, y_path):
    def __init__(self):
        self.x_list = glob.glob(x_path + "*.jpg")
        self.y_list = glob.glob(y_path + "*.npy")
        self.x_list.sort()
        self.y_list.sort()
        self.trainset = [(a,b) for a,b in zip(self.x, self.y)]
        self.i = 0

    def next_batch(self, batch_size = 8):
        if((i+1)*batch_size >= len(trainset)):
            i = 0
            random.shuffle(trainset)
        data = trainset[i* batch_size : (i+1)* batch_size]
        i = i + 1
        return data