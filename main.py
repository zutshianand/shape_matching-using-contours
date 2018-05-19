import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

someshapes = cv2.imread('./images/someshapes.jpg')
someshapes_original = someshapes.copy()

someshapes = cv2.cvtColor(someshapes , cv2.COLOR_BGR2GRAY)
somesmoothened = cv2.GaussianBlur(someshapes , (5 , 5) , 0)
ret , th_someshapes = cv2.threshold(somesmoothened , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
some_shapes , contours , hierarchy = cv2.findContours(th_someshapes , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours , key = cv2.contourArea , reverse = True)
contours = contours[1:]
#print(len(contours))

for(i,c) in enumerate(contours):
    epsilon = 0.01 * cv2.arcLength(c , True)
    approx = cv2.approxPolyDP(c , epsilon , True)
    cv2.drawContours(th_someshapes , [approx] , 0 , (170,255,0) , 4)
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
#    cv2.putText(th_someshapes , str(i + 1) , (cx , cy) , cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , 1 , (255,130,0) , 2 , cv2.LINE_AA)


#cv2.imshow('bunch of shapes' , th_someshapes)
#cv2.imwrite('bunch_of_shapes.jpg', th_someshapes)
#cv2.waitKey()
#cv2.destroyAllWindows()

bunch_of_shapes = cv2.imread('./images/bunchofshapes.jpg')
bunch_of_shapes_original = bunch_of_shapes.copy()
bunch_of_shapes = cv2.cvtColor(bunch_of_shapes , cv2.COLOR_BGR2GRAY)
ret , th_bunch_of_shapes = cv2.threshold(bunch_of_shapes , 230 , 255 , cv2.THRESH_BINARY)
image_th_bunch_of_shapes , contours_th_bunch_of_shapes , hierarchy_th_bunch_of_shapes = cv2.findContours(th_bunch_of_shapes ,
                                                                                                         cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
contours_th_bunch_of_shapes = sorted(contours_th_bunch_of_shapes , key = cv2.contourArea , reverse = True)
contours_th_bunch_of_shapes = contours_th_bunch_of_shapes[1:]

for(i,c) in enumerate(contours_th_bunch_of_shapes):
    epsilon = 0.01 * cv2.arcLength(c , True)
    contours_th_bunch_of_shapes[i] = cv2.approxPolyDP(c , epsilon , True)
    cv2.drawContours(bunch_of_shapes_original , [contours_th_bunch_of_shapes[i]] , 0 , (0,255,0) , 3)


#cv2.imshow('bunch of shapes' , bunch_of_shapes_original)
#cv2.imwrite('bunch_of_shapes_original.jpg' , bunch_of_shapes_original)
#cv2.waitKey()
#cv2.destroyAllWindows()

for(i , c1) in enumerate(contours):
    epsilon1 = 0.01 * cv2.arcLength(c1 , True)
    approx1 = cv2.approxPolyDP(c1 , epsilon1 , True)

    # match approx1 to all contours in bunch_of_shapes_contours
    res = 1000000
    indx = -1
    for(j , c2) in enumerate(contours_th_bunch_of_shapes):
        value = cv2.matchShapes(approx1 , c2 , 1 , 0.0)
        if value < 0.1 and value < res:
            res = value
            indx = j

    #contour number j is matching to i
    if indx != -1:
        M = cv2.moments(contours_th_bunch_of_shapes[indx])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(bunch_of_shapes_original, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('bunch of shapes after matching' , bunch_of_shapes_original)
#cv2.imwrite('bunch_of_shapes_after_matching.jpg' , bunch_of_shapes_original)
cv2.waitKey()
cv2.destroyAllWindows()
