#!/usr/bin/env python3
from glob import glob

def resize(height, width, image_path):
    return resize_image

def onehot(image):
    return numpy

if __name__ == '__main__':
    image_path = ''
    label_path = ''

    images = glob('%s/*.png' % image_path)
    labels = glob('%s/*.png' % label_path)

    for image_path, label_path in zip(images, labels):
        image = resize(image_path)
        label = resize(label_path)
        label = onehot(label)

        save_image(image)
        save_numpy(label)
