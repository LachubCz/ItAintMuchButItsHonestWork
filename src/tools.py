import os
import cv2

from image import Image

def parse_data(filename, images_path):
    with open(filename) as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    data = []
    for i, item in enumerate(content):
        if i == 0:
            continue
        parametres = item.split(',')
        img = Image(parametres[0], parametres[1], parametres[2], parametres[3], 
                    parametres[4], parametres[5], parametres[6], parametres[7], 
                    parametres[8])
        data.append(img)

    return data
