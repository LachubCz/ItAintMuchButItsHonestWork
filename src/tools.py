import os
import cv2
import numpy as np

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

        image = cv2.imread(os.path.join(data_path, parametres[0]), -1)
        image_processed = image * np.uint16(65535.0 / max(image.ravel()))

        ground_truth = cv2.imread(os.path.join(ground_truths_path, parametres[0][:parametres[0].rfind('.')] + ".png"), -1)
        ground_truth_processed = np.copy(ground_truth)
        indices = np.where(np.any(ground_truth_processed != [0, 0, 255], axis = -1))
        ground_truth_processed[indices] = [0, 0, 0]
        indices = np.where(np.all(ground_truth_processed == [0, 0, 255], axis = -1))
        ground_truth_processed[indices] = [255, 255, 255]
        ground_truth_processed = cv2.cvtColor(ground_truth_processed, cv2.COLOR_BGR2GRAY)
        
        img = Image(image, image_processed, ground_truth, ground_truth_processed,
                    parametres[0], parametres[1], parametres[2], parametres[3], 
                    parametres[4], parametres[5], parametres[6], parametres[7], 
                    parametres[8])
        data.append(img)

    return data

if __name__ == '__main__':
    parse_data("./data/ground_truths_develop.csv", "./data/images/", "./data/ground_truths/")