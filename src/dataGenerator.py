#!/usr/bin/python3

# example of zoom image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

import sys
import cv2

#usage:
# ./dataGenerator.py filename data_path grand_truths_path augment_dir
# filename = sys.argv[1]
# dataPath = sys.argv[2]
# grandPath = sys.argv[3]
# augmentDir = sys.argv[4]

#if augment dir is not exists -> create


# load the image
img = load_img('1.tiff')
# y1 = load_img('my1.png')
# x2 = load_img('my2.png')
# y2 = load_img('my3.png')
# # convert to numpy array
data = img_to_array(img)
#data_png = cv2.imread('1.png')
#data_tif = cv2.imread('1.tiff')

# expand dimension to one sample
y = expand_dims(data_png, 0)
x = expand_dims(data_tif, 0)
print(x.shape)
print(y.shape)
h, w, _ = x[0].shape
x[0,0,:,:] = 0
x[0,h-1,:,:] = 0
x[0,:,0,:] = 0
x[0,:,w-1,:] = 0

y[0,0,:,:] = 0
y[0,h-1,:,:] = 0
y[0,:,0,:] = 0
y[0,:,w-1,:] = 0
# create image data augmentation generator
shift = 0.2
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                             zoom_range=[0.5,1.0], rotation_range=90, 
                             width_shift_range=[-shift,shift], height_shift_range=[-shift,shift])
# prepare iterator
it = datagen.flow(x, y, batch_size=1)
# generate samples and plot
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    #print(batch[0].shape)
    print(batch[1].shape)
    # convert to unsigned integers for viewing
    image = batch[1][0].astype('uint32')
    # plot raw pixel data
    pyplot.imshow(image)
# show the figure
pyplot.show()