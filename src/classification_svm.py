from os import listdir
from os.path import isfile, join

import cv2
import numpy as np

from extracting_inception import create_graph, extract_features
from train_svm import train_svm_classifer

from tools import parse_data

if __name__ == '__main__':
    batch = 1000
    filenames = ["./images_png/"+f for f in listdir("./images_png") if isfile(join("./images_png", f))]

    create_graph("./tensorflow_inception_graph.pb")

    labels = []
    for i, item in enumerate(filenames[:batch]):
        if item[-5] == 'T':
            labels.append(1)
        else:
            labels.append(0)

    features = extract_features(filenames[:batch], verbose=True)

    train_svm_classifer(features, labels, "model.pkl")