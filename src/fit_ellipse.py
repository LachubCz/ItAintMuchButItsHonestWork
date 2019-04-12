import os
import cv2
import numpy as np

def fit_ellipse(original, segmented):
    # find contours
    contours, hierarchy = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # fitting ellipse
    bounding_rect = None
    for i in range(len(contours)):
        if len(contours[i]) > 50:
            bounding_rect = cv2.fitEllipse(contours[i])
            break
    if bounding_rect is None:
        print("No ellipse found")
        return

    # draw ellipse
    test = cv2.cvtColor(np.uint8(np.clip(original, 0, 255)), cv2.COLOR_GRAY2BGR)
    cv2.ellipse(test, bounding_rect, (0, 0, 255), 5)

    # count parameters
    ellipse_center_x = bounding_rect[0][0]
    ellipse_center_y = bounding_rect[0][1]
    ellipse_majoraxis = max(bounding_rect[1]) / 2.0
    ellipse_minoraxis = min(bounding_rect[1]) / 2.0
    ellipse_angle = bounding_rect[2]

    # print them
    # print(ellipse_center_x)
    # print(ellipse_center_y)
    # print(ellipse_majoraxis)
    # print(ellipse_minoraxis)
    # print(ellipse_angle)

    # show image
    cv2.imshow("test", test)
    cv2.waitKey(0)

if __name__ == '__main__':
    with open("./data/ground_truths_develop.csv") as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    data = []
    for i, item in enumerate(content):
        if i == 0:
            continue
        parametres = item.split(',')
        image = cv2.imread(os.path.join("./data/images/", parametres[0]), -1)
        # threshold
        ret, thresh = cv2.threshold(image, 127, 255, 0)
        thresh = np.uint8(np.clip(thresh, 0, 255))
        fit_ellipse(image, thresh)


# draw minimal rectangle
# box = cv2.boxPoints(bounding_rect)
# box = np.int0(box)
# test = cv2.drawContours(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), [box], -1, (0,255,0), 3)
# test = cv2.circle(test, (int(bounding_rect[0][0]), int(bounding_rect[0][1])), 5, (0, 0, 255), 5)
# test = cv2.circle(test, (int(bounding_rect[1][0]), int(bounding_rect[1][1])), 5, (0, 0, 255), 5)