import os
from os import listdir
from os.path import isfile, join
import sys 
import argparse

import cv2
import numpy as np

import matplotlib.pyplot as plt

from tools import parse_data
from ellipse_fit_evaluation import evaluate_ellipse_fit, __get_gt_ellipse_from_csv
from fit_ellipse import fit_ellipse

def get_args():
    """
    method for parsing of arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--mode", action="store", default=["eval", "entry"],
                        help="application mode")
    parser.add_argument('--csv-input', action="store", 
                        help='Text file with line names and transcripts for training.')
    parser.add_argument('--csv-output', action="store", 
                        help='Text file with line names and transcripts for training.')
    parser.add_argument('--images-path', action="store", required=True, 
                        help='Text file with line names and transcripts for training.')
    parser.add_argument('--ground-truths-path', action="store", 
                        help='Text file with line names and transcripts for training.')

    args = parser.parse_args()

    if args.mode == "eval":
        if not os.path.isfile(args.csv_input):
            print("CSV input file doesn't exist.")
            sys.exit(-1)
        if not os.path.isdir(args.images_path):
            print("Images path is not valid.")
            sys.exit(-1)
    elif args.mode == "entry":
        if not os.path.isdir(args.images_path):
            print("Images path is not valid.")

    return args


def perf_measure(y_actual, y_true):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_true)): 
        if y_actual[i] !=0 and y_true[i] == 1:
           TP += 1
        if y_true[i] == 1 and y_actual[i] == 0:
           FP += 1
        if y_actual[i] == 0 and y_true[i] == 0:
           TN += 1
        if y_true[i] == 0 and y_actual[i] != 0:
           FN += 1

    score =  {
      "TP": TP,
      "FP": FP,
      "TN": TN,
      "FN": FN
    }

    return score


def get_scores(image_filename, fit_ellipse, csv_filepath):
    gt_ellipse = __get_gt_ellipse_from_csv(image_filename, csv_filepath)

    if gt_ellipse:
        if fit_ellipse:
            return 1,1
        else:
            return 0,1
    else:
        if fit_ellipse:
            return 0,0
        else:
            return 1,0


if __name__ == '__main__':
    args = get_args()
    data = parse_data(args.csv_input, args.images_path, args.ground_truths_path)

    if args.mode == "eval":
        score_sum = 0
        true_values = []
        predicted_values = []
        for i, item in enumerate(data):
            ret, thresh = cv2.threshold(item.image, 127, 255, 0)
            thresh = np.uint8(np.clip(thresh, 0, 255))

            ellipse = fit_ellipse(item.image, thresh)

            score = evaluate_ellipse_fit(item.filename, ellipse, args.csv_input)

            predicted_value, true_value = get_scores(item.filename, ellipse, args.csv_input)
            true_values.append(true_value)
            predicted_values.append(predicted_value)
            score_sum += score

            print("{} - {}" .format(item.filename, (round(score, 2))))

        print("Overall score: {}/{} = {}%" .format((round(score_sum, 2)), len(data), round((score_sum/len(data))*100, 2)))

        dictionary = perf_measure(predicted_values, true_values)
        print("TP", dictionary["TP"])
        print("FP", dictionary["FP"])
        print("TN", dictionary["TN"])
        print("FN", dictionary["FN"])

        plt.bar(list(dictionary.keys()), dictionary.values(), color='g')
        plt.show()
    elif args.mode == "entry":
        filenames = [f for f in listdir(args.images_path) if isfile(join(args.images_path, f))]
