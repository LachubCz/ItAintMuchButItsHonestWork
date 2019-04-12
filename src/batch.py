#!/usr/bin/python3

import numpy as np
from image import Image
from tools import parse_data
import cv2
import random
from datetime import datetime

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot


class DataSet(object):
    def __init__(self, ImageList):
        self.Images = ImageList
        self.p_images, self.labels, self. grand_truths = DataSet.strip_futilities(ImageList)
        random.seed(datetime.now())

    @staticmethod
    def augmentImage(imageArr, grandArr, batchSize):
        
        for imgs in [imageArr, grandArr]:
            for img in imgs:
                h, w = img[0].shape
                img[0,0,:] = 0
                img[0,h-1,:] = 0
                img[0,:,0] = 0
                img[0,:,w-1] = 0

        shift = 0.2
        data_gen_args = dict(   data_format="channels_first",
                                rotation_range=90,
                                height_shift_range=[-shift,shift],
                                width_shift_range=[-shift,shift],
                                zoom_range=0.2,
                                horizontal_flip=True,
                                vertical_flip=True)

        img_dataGen = ImageDataGenerator(**data_gen_args)
        g_tr_dataGen = ImageDataGenerator(**data_gen_args)

        seed = random.randint(0,65535)
        img_gen = img_dataGen.flow(imageArr, seed=seed, batch_size=batchSize)
        g_tr_gen = g_tr_dataGen.flow(grandArr, seed=seed, batch_size=batchSize)

        new_img = img_gen.next()
        new_g_t = g_tr_gen.next()
        new_img = np.uint16(new_img)
        new_g_t = np.uint16(new_g_t)
        
        return new_img, new_g_t
        
        
    @staticmethod
    def strip_futilities(data):
        images = []
        labels = []
        grand_truths = []
        for i, item in enumerate(data):
            images.append(np.array([item.processed_image]))
            labels.append(np.array([item.ellipse]))
            grand_truths.append(np.array([item.processed_ground_truths]))

        return np.array(images), np.array(labels), np.array(grand_truths)


    def getBatch(self, batchSize, isClassNet=True):
        img, g_truths = DataSet.augmentImage(self.p_images, self.grand_truths, batchSize)
        labels = []
        
        if batchSize > len(self.p_images):
            while len(img) != batchSize:
                curSize = batchSize - len(img)
                img2, g_truths2 = DataSet.augmentImage(self.p_images, self.grand_truths, curSize)
                img = np.concatenate((img, img2), axis=0)
                g_truths = np.concatenate((g_truths, g_truths2), axis=0)

        #get labels
        for i in range(len(img)):
            unique, counts = np.unique(g_truths[i], return_counts=True)
            if unique[-1] > 60000:
                labels.append(np.array([1]))
            else:
                labels.append(np.array([0]))

        if isClassNet:
            return img, np.array(labels)
        else:
            return img, g_truths


if __name__ == "__main__":
    trn_data = parse_data("../data/ground_truths_develop.csv", "../data/images/", "../data/ground_truths/")
    myData = DataSet(trn_data)
    x, y = myData.getBatch(100) 
    print(x.shape, y.shape)