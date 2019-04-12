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
        random.seed(datetime.now())

    @staticmethod
    def augmentImage(Image):
        p_img = np.expand_dims(Image.processed_image, 0)
        p_g_tr = np.expand_dims(Image.processed_ground_truths, 0)

        print(p_img.shape)
        print(p_g_tr.shape)

        for img in [p_img, p_g_tr]:
            h, w = img[0].shape
            img[0,0,:] = 0
            img[0,h-1,:] = 0
            img[0,:,0] = 0
            img[0,:,w-1] = 0

        curShape = p_img.shape
        p_img = np.reshape(p_img, (curShape[0], curShape[1], curShape[2], 1))
        p_g_tr = np.reshape(p_g_tr, (curShape[0], curShape[1], curShape[2], 1))


        data_gen_args = dict(   rotation_range=90,
                                height_shift_range=[-0.2,0.2],
                                zoom_range=[0.5,1.0],
                                horizontal_flip=True,
                                vertical_flip=True)

        img_dataGen = ImageDataGenerator(**data_gen_args)
        g_tr_dataGen = ImageDataGenerator(**data_gen_args)

        seed = random.randint(0,65535)
        img_gen = img_dataGen.flow(p_img, seed=seed)
        g_tr_gen = g_tr_dataGen.flow(p_g_tr, seed=seed)

        new_img = img_gen.next()[0]
        new_g_t = g_tr_gen.next()[0]

        new_img = np.uint16(new_img)
        new_g_t = np.uint16(new_g_t)

        
        cv2.imshow("muj",Image.processed_ground_truths)
        cv2.waitKey(0)
        cv2.imshow("muj_new", new_img)
        cv2.waitKey(0)
        cv2.imshow("muj_new_g", new_g_t)
        cv2.waitKey(0)
        
        


    def getBatchClas(self, batchSize):
        pass




if __name__ == "__main__":
    trn_data = parse_data("../data/ground_truths_develop.csv", "../data/images/", "../data/ground_truths/")
    myData = DataSet(trn_data)
    myData.augmentImage(random.choice(trn_data))