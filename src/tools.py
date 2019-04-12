import os
import cv2

from image import Image

def parse_data(filename, data_path, ground_truths_path):
    with open(filename) as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    data = []
    for i, item in enumerate(content):
        if i == 0:
            continue
        parametres = item.split(',')
        image = cv2.imread(os.path.join(data_path, parametres[0]))
        ground_truth = cv2.imread(os.path.join(ground_truths_path, parametres[0]))
        img = Image(image, ground_truth, 
                    parametres[0], parametres[1], parametres[2], parametres[3], 
                    parametres[4], parametres[5], parametres[6], parametres[7], 
                    parametres[8])
        data.append(img)

    return data

if __name__ == '__main__':
    parse_data("./data/ground_truths_develop.csv", "./data/images/", "./data/ground_truth/")