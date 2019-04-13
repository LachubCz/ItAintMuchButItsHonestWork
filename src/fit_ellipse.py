import os
import cv2
import numpy as np
import math

def fitting_error(contour):
    # fit ellipse
    bounding_rect = cv2.fitEllipse(contour)

    # fit line
    line = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    p1 = (line[2][0], line[3][0])
    p2 = (p1[0] + line[0][0], p1[1] + line[1][0])

    # find cumulative difference
    cumulative_diff = 0
    for j in range(len(contour)):
        p0 = contour[j][0]
        top = abs((p2[1] - p1[1]) * p0[0] - (p2[0] - p1[0]) * p0[1] + p2[0] * p1[1] - p2[1] * p1[0])
        bottom = pow((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2, 0.5)
        cumulative_diff += top / bottom
    
    # its a line! exterminate it!
    # print(min(bounding_rect[1]) / 2.0)
    if cumulative_diff < 25 - math.log(min(bounding_rect[1]) * 10):
        # print(min(bounding_rect[1]) / 2.0)
        return 100000

    # get its focal points
    a = max(bounding_rect[1]) / 2.0
    b = min(bounding_rect[1]) / 2.0
    f = pow(a**2 - b**2, 0.5)
    v_f = (math.cos(bounding_rect[2]) * f, math.sin(bounding_rect[2]) * f)
    f1 = (bounding_rect[0][0] - v_f[0], bounding_rect[0][1] - v_f[1])
    f2 = (bounding_rect[0][0] + v_f[0], bounding_rect[0][1] + v_f[1])
    f = pow(a**2 - b**2, 0.5)
    a2 = max(bounding_rect[1])

    # test
    # v_t = (math.cos(bounding_rect[2]) * a, math.sin(bounding_rect[2]) * a)
    # p_t = (bounding_rect[0][0] - v_t[0], bounding_rect[0][1] - v_t[1])
    # v1 = (p_t[0] - f1[0], p_t[1] - f1[1])
    # v2 = (p_t[0] - f2[0], p_t[1] - f2[1])
    # v1s = pow(v1[0]**2 + v1[1]**2, 0.5)
    # v2s = pow(v2[0]**2 + v2[1]**2, 0.5)
    # if abs(v1s + v2s - a2) > 0.001:
    #     print(abs(v1s + v2s - a2))
    
    # find cumulative difference
    cumulative_diff = 0
    for j in range(len(contour)):
        v1 = (contour[j][0][0] - f1[0], contour[j][0][1] - f1[1])
        v2 = (contour[j][0][0] - f2[0], contour[j][0][1] - f2[1])
        v1s = pow(v1[0]**2 + v1[1]**2, 0.5)
        v2s = pow(v2[0]**2 + v2[1]**2, 0.5)
        cumulative_diff += abs(v1s + v2s - a2)
    return cumulative_diff

def recursive_contour_divide(contour):
    err = fitting_error(contour)
    # print(err)
    if err < 400:
        return contour
    if err > 99999:
        return None
    
    if len(contour) > 60:
        half1 = recursive_contour_divide(contour[:int(len(contour) / 2)])
        half2 = recursive_contour_divide(contour[int(len(contour) / 2):])
        if half1 is not None and half2 is not None:
            return np.concatenate((half1, half2), axis = 0)
        if half1 is not None:
            return half1
        if half2 is not None:
            return half2
    
    return None

def fit_ellipse(original, segmented, file_to_open = None):
    # find contours
    #_, contours, _ = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # windows shit
    contours, _ = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # this is how we do it
    #print("cont:", np.shape(contours))

    # get max length contour
    new_cont = np.array([])
    if len(contours) > 0:
        new_cont = contours[0]
        for i in range(len(contours)-1):
            new_cont = np.concatenate((new_cont, contours[i]), axis=0)
    
    # no ellipse found
    if len(new_cont) < 5:
        # print("Less than five")
        return None

    # get reduced contour
    #reduced_contour = recursive_contour_divide(new_cont)
    reduced_contour = new_cont
    # its bullshit, i did not hit her, i did not. oh hi mark!
    if reduced_contour is None:
        # print("Reduced is none")
        reduced_contour = new_cont

    # fit ellipse
    bounding_rect = cv2.fitEllipse(reduced_contour)

    # draw ellipse
    test = cv2.cvtColor(np.uint8(np.clip(segmented, 0, 255)), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(test, reduced_contour, -1, (0,255,0), 3)
    cv2.ellipse(test, bounding_rect, (0, 0, 255), 5)

    # count parameters
    ellipse_center_x = bounding_rect[0][0]
    ellipse_center_y = bounding_rect[0][1]
    ellipse_majoraxis = max(bounding_rect[1]) / 2.0
    ellipse_minoraxis = min(bounding_rect[1]) / 2.0
    ellipse_angle = bounding_rect[2] + 90

    # print them
    # print(ellipse_center_x)
    # print(ellipse_center_y)
    # print(ellipse_majoraxis)
    # print(ellipse_minoraxis)
    # print(ellipse_angle)

    # show image
    if __name__ == '__main__':
        cv2.imshow("result", cv2.resize(test, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA))
        if file_to_open is not None:
            ref = cv2.imread(file_to_open, -1)
            cv2.imshow("reference", cv2.resize(ref, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA))
        cv2.waitKey(0)

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
        maxval = max(image.ravel())
        new_image =  np.uint8(np.clip(255/maxval * image, 0, 255))

        # threshold
        # k_size = 3
        # kernel = np.ones((k_size, k_size),np.float32) / k_size**2
        # dst = cv2.filter2D(new_image,-1,kernel)
        #tmpImg = cv2.fastNlMeansDenoisingColored(cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR), h = 5, templateWindowSize = 5, searchWindowSize = 15)
        #blur = cv2.GaussianBlur(cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY),(5,5),0)
        blur = cv2.GaussianBlur(new_image, (5, 5), 0)

        # blur
        #ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret, thresh = cv2.threshold(blur, 80, 255, 0)
        thresh = np.uint8(np.clip(thresh, 0, 255))


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
        res = fit_ellipse(new_image, thresh, os.path.join("./data/ground_truths/", parametres[0][:parametres[0].rfind('.')] + ".png"))
        if res is None:
            print(parametres[0])

    # draw minimal rectangle
    # box = cv2.boxPoints(bounding_rect)
    # box = np.int0(box)
    # test = cv2.drawContours(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), [box], -1, (0,255,0), 3)
    # test = cv2.circle(test, (int(bounding_rect[0][0]), int(bounding_rect[0][1])), 5, (0, 0, 255), 5)
    # test = cv2.circle(test, (int(bounding_rect[1][0]), int(bounding_rect[1][1])), 5, (0, 0, 255), 5)