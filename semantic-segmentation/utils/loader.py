import glob
import random

class DataLoader():

    def __init__(self, x_path, y_path):
        self.x_list = glob.glob(x_path + "/*.png")
        self.y_list = glob.glob(y_path + "/*.npy")
        self.x_list.sort()
        self.y_list.sort()
        self.trainset = [(a,b) for a,b in zip(self.x_list, self.y_list)]
        self.i = 0

    def next_batch(self, batch_size = 8):
        if((self.i + 1) * batch_size >= len(self.trainset)):
            self.i = 0
            random.shuffle(self.trainset)

        data = self.trainset[self.i * batch_size : (self.i + 1) * batch_size]
        self.i = self.i + 1

        return data
