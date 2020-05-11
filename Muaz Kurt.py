#!/usr/bin/env python3


# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                            -min(0, x1), max(x2 - img.shape[1], 0),cv.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2

def imcrop(img, bbox):
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2]



# C# program to find convex hull of a set of points. Refer  
# https://www.geeksforgeeks.org/orientation-3-ordered-points/ 
# for explanation of orientation() 
  
def Left_index(points): 
      
    ''' 
    Finding the left most point 
    '''
    minn = 0
    for i in range(1,len(points)): 
        if points[i][0] < points[minn][0]: 
            minn = i 
        elif points[i][0] == points[minn][0]: 
            if points[i][1] > points[minn][1]: 
                minn = i 
    return minn 
  
def orientation(p, q, r): 
    ''' 
    To find orientation of ordered triplet (p, q, r).  
    The function returns following values  
    0 --> p, q and r are colinear  
    1 --> Clockwise  
    2 --> Counterclockwise  
    '''
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]) 
    if val == 0: 
        return 0
    elif val > 0: 
        return 1
    else: 
        return 2
  
'''
    Finds the conwex hull points of the given point cloud
'''
def convexHull(points): 
      
    # There must be at least 3 points  
    n = len(points)
    if(n < 4):
        return points
    # Find the leftmost point 
    l = Left_index(points) 

    hull = []
    
    
    p = l 
    q = 0
    max_iter = 20
    while(max_iter > 0): 
        max_iter -= 1
        # Add current point to result
        hull.append(points[p]) 
  
        q = (p + 1) % n 
  
        for i in range(n): 
            # If i is more counterclockwise  
            # than current q, then update q  
            if(orientation(points[p][0], points[i][0], points[q][0]) == 2): 
                q = i 
  
        p = q 
  
        # While we don't come to first point 
        if(p == l): 
            break

    i = 0
    '''
    while (i < len(hull) - 1):
        if((hull[i][0][0] - hull[i + 1][0][0]) ** 2 + (hull[i][0][1] - hull[i + 1][0][1]) < 3 ** 2):
            hull.pop(i)
            i -= 1
        i += 1
    '''
    return hull


'''
    Takes two line descriptors as Rho, Theta pairs.
    Then calculates their line equations as homogenious form.
    Cross products them and gets their intersection point.
    If the result's 3rd row is 0, then these line intersection is in the infinity.
        So these lines are parallel to each other.
'''
def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    constant = 1000
    if(np.abs(theta1 - theta2) > 3 * np.pi / 180):
        a1, b1 = np.cos(theta1), np.sin(theta1)
        a2, b2 = np.cos(theta2), np.sin(theta2)
        x0, y0, x1, y1 =  (a1 * rho1 + constant * -b1), (b1 * rho1 + constant * a1),  (a1 * rho1 - constant * -b1), (b1 * rho1 - constant * a1)
        x2, y2, x3, y3 = (a2 * rho2 + constant * -b2), (b2 * rho2 + constant * a2),  (a2 * rho2 - constant * -b2), (b2 * rho2 - constant * a2)
        l1 = np.cross([x0, y0, 1], [x1, y1, 1])
        l2 = np.cross([x2, y2, 1], [x3, y3, 1])
        cross_p = np.cross(l1, l2)
        if(cross_p[2] != 0):
            return [int(cross_p[0] / cross_p[2]), int(cross_p[1] / cross_p[2])]
    return None


'''
    Takes two arrays.
        Lines, each entry describes a line in rho-theta form
        intersection, output will be put in here
    Calculates each lines intersection to each other.
'''
def find_intersections(lines, intersections):
    for i in range(len(lines)):
        l = lines[i][0]
        for j in range(i, len(lines)):
            l2 = lines[j][0]
            temp = intersection(lines[i], lines[j])
            if(temp != None and temp[0] > -1 and temp[1] > -1 and temp[0] <= 800 and temp[1] <= 600):
                intersections.append((temp, ()))
    return intersections
    

'''
    Takes an array and a distance margin.
    Calculates mean of each array entry in between mean_distance far to each other.
    Updates the point array and returns.
'''
def filter_mean_cross(intersections, mean_distance = 30):
    intersections = np.array(intersections)
    i = 0
    while (i < len(intersections)):
        local = [intersections[i][0][0], intersections[i][0][1]]
        counted = 1
        j = i + 1
        while (j < len(intersections)):
            if((intersections[i][0][0] - intersections[j][0][0]) ** 2 + (intersections[i][0][1] - intersections[j][0][1]) ** 2 < mean_distance ** 2):
                local[0] += intersections[j][0][0]
                local[1] += intersections[j][0][1]
                intersections = np.delete(intersections, j, 0)
                counted += 1
                j -= 1
            j += 1
        intersections[i][0] = int(local[0] / counted), int(local[1] / counted)
        i += 1
    return intersections
    

'''
    Clears all points far away from the mean of the intersection points
'''
def filter_onthe_line(intersections):
    start_count = len(intersections)
    if(start_count > 0):
        mean_coordinate = [0, 0]
        for i in intersections:
            mean_coordinate[0] += i[0][0]
            mean_coordinate[1] += i[0][1]
        mean_coordinate[0] /= len(intersections)
        mean_coordinate[1] /= len(intersections)
        i = 0
        while (i < len(intersections)):
            x1, y1 = intersections[i][0]
            if(np.abs(x1 - mean_coordinate[0]) > 300 or np.abs(y1 - mean_coordinate[1]) > 200):
                intersections = np.delete(intersections, i, 0)
            else: 
                intersections[i][1] = (np.abs(intersections[i][0][0] - mean_coordinate[0]), np.abs(intersections[i][0][1] - mean_coordinate[1]))
                i += 1

    return intersections

'''
    Finds the convex hull of the given intersection points and returns its four corner.
'''
def find_corner(intersections):
    constant = 20
    corner_left_up = (320, 240)
    corner_left_down = (320, 0)
    corner_right_up = (0, 240)
    corner_right_down = (0, 0)
    intersections = convexHull(intersections)
    if(len(intersections) > 3):
        corner_left_down    = intersections[0][0]
        corner_left_up      = intersections[1][0]
        corner_right_up     = intersections[2][0]
        corner_right_down   = intersections[3][0]
        
        for point in intersections:
            x0, y0 = point[0]
            #print(x0, y0)
            if(x0 - constant < corner_left_up[0] and y0 - constant < corner_left_up[1]):
                corner_left_up = point[0]
        for point in intersections:
            x0, y0 = point[0]
            if (x0 - constant < corner_left_down[0] and y0 + constant > corner_left_down[1]):
                corner_left_down = point[0]
        for point in intersections:
            x0, y0 = point[0]
            if (x0 + constant > corner_right_up[0] and y0 - constant < corner_right_up[1]):
                corner_right_up = point[0]
        for point in intersections:
            x0, y0 = point[0]
            if (x0 + constant > corner_right_down[0] and y0 + constant > corner_right_down[1]):
                corner_right_down = point[0]
    return np.array([corner_left_up, corner_left_down, corner_right_up, corner_right_down])
    

'''
    Takes an image and calculates four corner points of the rectangle in it.
    Returns the corner points.
'''
def calculate_rectangle_corners(img, main_i = 0):
    view_before_filter = img.copy()
    view_after_filter = img.copy()
    grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    grey = cv.GaussianBlur(grey, (17, 17), 0)
    
    adapt_type = cv.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv.THRESH_BINARY_INV
    grey = cv.adaptiveThreshold(grey, 255, adapt_type, thresh_type, 21, 3)

    edges = cv.Canny(grey,100, 180, apertureSize = 3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is not None:
        intersections = []
        find_intersections(lines, intersections)
        if(intersections != None):
            org = len(intersections)
            intersections = filter_mean_cross(intersections)
            for temp in intersections:
                cv.circle(view_before_filter, (temp[0][0], temp[0][1]), 10, (0, 255, 0), -1)
           
            first = len(intersections)

            l = list(intersections)
            l.sort(key= lambda x : -x[0][1])
            intersections = np.array(l)
            if(main_i == 0):
                intersections = filter_onthe_line(intersections)
            for a in intersections:
                cv.circle(view_before_filter, (a[0][0], a[0][1]), 5, (250, 0, 0), -1)

            arr = convexHull(intersections)
            for i in range(len(arr)):
                a = arr[i]
                b = arr[(i + 1) % len(arr)]
                #cv.circle(view_before_filter, a[0], 3, (0, 0, 255), -1)
                cv.line(view_before_filter, a[0], b[0], (255, 255, 0), 2)
            
            arr = find_corner(intersections)
            for a in arr:
                cv.circle(view_after_filter, (a[0], a[1]), 5, (0, 0, 255), -1)
            cv.imshow("before Filter", view_before_filter)
            cv.imshow("after_filter", view_after_filter)
            return arr
    return np.array([])

 


def main():
    cv.namedWindow("Video")
    cv.namedWindow("before Filter")
    cv.namedWindow("after_filter")
    img = cv.imread("./soccer fieild.jpg")
    
    #wrap = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_WRAP)
    img = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_CONSTANT, value=(255,255,255))
    base_arr = np.array(calculate_rectangle_corners(img, 1))
    ref = img.copy()
    
    '''for temp in base_arr:
        cv.circle(img, temp, 3, (0, 255, 0), -1)
    '''
    cv.imshow("ref", img)
    
    '''
    cap = cv.VideoCapture("/dev/video0")
    cap = cv.VideoCapture("/dev/video2")
    cap = cv.VideoCapture("2020-04-13-201559.webm")
    '''
    cap = cv.VideoCapture("2020-04-16-020633.webm")
    ch = cv.waitKey(200)

    while True:
        _status, img = cap.read()
        if(_status):
            arr_temp = calculate_rectangle_corners(img)
            if(len(arr_temp) == 4):
                h, status = cv.findHomography(arr_temp, base_arr)  
                if(status[0][0] == 1 and 
                    status[1][0] == 1 and 
                    status[2][0] == 1 and 
                    status[3][0] == 1
                    ):
                    oput = cv.warpPerspective(img, h, (ref.shape[1], ref.shape[0]))
                    cv.imshow("output", oput)
            cv.imshow("Video", img)
            
            ch = cv.waitKey(1)
            if ch == 27:
                break
        else:
            break


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
