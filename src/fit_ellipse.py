import os
import cv2
import numpy as np

def fit_ellipse(original, segmented):
    # find contours
    _, contours, _ = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print("cont:", np.shape(contours))
    # fitting ellipse
    bounding_rect = None
    for i in range(len(contours)):
        if len(contours[i]) > 50:
            bounding_rect = cv2.fitEllipse(contours[i])
            break
    if bounding_rect is None:
        #print("No ellipse found")
        return None

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
    #cv2.imshow("test", test)
    #cv2.waitKey(0)

    ellipse =  {
      "center": [ellipse_center_x, ellipse_center_y],
      "axes": [ellipse_majoraxis, ellipse_minoraxis],
      "angle": ellipse_angle
    }

    return ellipse


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
        print(image.dtype, np.shape(image))
        ret, thresh = cv2.threshold(image, 127, 255, 0)
        thresh = np.uint8(np.clip(thresh, 0, 255))
        cv2.imwrite("{}.png" .format(i), thresh)

        ####################

        #kernel = np.ones((3, 3), np.uint8) 
        #closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 
        #                            kernel, iterations = 2) 
        #  
        ## Background area using Dialation 
        #bg = cv2.dilate(closing, kernel, iterations = 1) 
        #  
        ## Finding foreground area 
        #dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0) 
        #ret, fg = cv2.threshold(dist_transform, 0.02
        #                        * dist_transform.max(), 255, 0) 
        ##############################
        #image = cv2.imread(os.path.join("./data/images/", parametres[0]), cv2.CV_8UC1)

        #th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,0.5)
        #cv2.imwrite("{}_.png" .format(i), th3)

        #th2 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,0.5)
        #cv2.imwrite("{}__.png" .format(i), th2)

        #kernel = np.ones((3, 3), np.uint8) 
        #closing = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, 
        #                            kernel, iterations = 2) 
        #  
        ## Background area using Dialation 
        #bg = cv2.dilate(closing, kernel, iterations = 1) 
        #  
        ## Finding foreground area 
        #dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0) 
        #ret, fg = cv2.threshold(dist_transform, 0.02
        #                        * dist_transform.max(), 255, 0) 
        #cv2.imwrite("{}___.png" .format(i), fg)
        #cv2.imshow('image', fg) 
        #cv2.waitKey(0)
        ############
        #imwrite("{}.png" .format(i), thresh)
        #cv2.imshow("test", thresh)
        #cv2.waitKey(0)
        fit_ellipse(image, thresh)

    # draw minimal rectangle
    # box = cv2.boxPoints(bounding_rect)
    # box = np.int0(box)
    # test = cv2.drawContours(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), [box], -1, (0,255,0), 3)
    # test = cv2.circle(test, (int(bounding_rect[0][0]), int(bounding_rect[0][1])), 5, (0, 0, 255), 5)
    # test = cv2.circle(test, (int(bounding_rect[1][0]), int(bounding_rect[1][1])), 5, (0, 0, 255), 5)