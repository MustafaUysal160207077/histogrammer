import cv2
import numpy as np


class Camera:

    def __init__(self, cam_num):    #By using the "self" keyword we access the attributes and methods of the class in python.
    #__init__ expects a parameter (cam_num) . this is the argument that the VideoCapture of OpenCV requires.
    # If we wanted to have properties available right after the instantiation of a class, you had to include them in the __init__ method.
        self.cam_num = cam_num
        self.cap = None

    def open(self, width=640, height=480):
        self.cap = cv2.VideoCapture(self.cam_num, cv2.CAP_DSHOW)  # Capture a frame
        self.cap.set(3, width)  # propID os width is 3
        self.cap.set(4, height)  # set height
        if not self.cap.isOpened():
            print("Failed to open camera.")
            return

    def read(self):
        self.frame = cv2.imread('frame3.png')

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        return self.frame

    def find_biggest_contour(self):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_RGB2HSV)

        #h, s, v = cv2.split(hsv)

        lower_limit = np.array([0, 0, 93])
        upper_limit = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower_limit, upper_limit)  #created a mask to remove background
        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(self.frame, self.frame, mask=mask)
        fg = cv2.bitwise_and(self.frame, self.frame, mask=mask_inv)


        gray = cv2.cvtColor(bg,cv2.COLOR_RGB2GRAY)

       

        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1] #findContours function works better with binary images

        #cv2.imshow('image.png', thresh)
        #cv2.waitKey(0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) #remove noise
        #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


        cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1] #It accounts for the fact that different versions of OpenCV return different number of values for findContours().
        #https://stackoverflow.com/questions/55854810/opencv-version-4-1-0-drawcontours/55857838

        area_thresh = 10000
        cnt = 0

        for c in cntrs:
            area = cv2.contourArea(c)
            if area > area_thresh:
                #cv2.drawContours(self.frame, [c], 0, (255, 0, 0), 2)
                cnt = cnt + 1
                if cnt > 0:
                    rect = cv2.minAreaRect(c) #minArearect returns - ( center (x,y), (width, height), angle of rotation ).
                    box = cv2.boxPoints(rect) # The function finds the four vertices of a rotated rectangle.
                    #https: // docs.opencv.org / trunk / dd / d49 / tutorial_py_contour_features.html

                    box = np.int0(box) #converting numbers to integer
                    # crop image inside bounding box

                    centerX = rect[0][0]
                    centerY = rect[0][1]
                    W = rect[1][0] #width of contour
                    H = rect[1][1] #height of contour
                    Xs = [i[0] for i in box]
                    Ys = [i[1] for i in box]
                    x1 = min(Xs)
                    x2 = max(Xs)
                    y1 = min(Ys)
                    y2 = max(Ys)
                    angle = rect[2]
                    rotated = False
                    if angle < -45:
                        angle += 90
                        rotated = True
                    center = (round(centerX), round(centerY))
                    size = (int((x2 - x1)), int( (y2 - y1)))
                    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
                    #Calculates an affine matrix of 2D rotation.
                    cropped = cv2.getRectSubPix(self.frame, size, center) #crop contour

                    cropped = cv2.warpAffine(cropped, M, size) #rotate contour using 2D-RotationMatrix
                    croppedW = W if not rotated else H
                    croppedH = H if not rotated else W
                    self.image = cv2.getRectSubPix(
                        cropped, (int(croppedW ), int(croppedH)), (size[0] / 2, size[1] / 2)) #crop contour

                    hsv2 = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)


                    for x in range(0, self.image.shape[0]):
                        for y in range(0, self.image.shape[1]):
                            if hsv2[x, y, 0] > 50 or hsv2[x, y, 1] < 80 or hsv2[x, y, 2] > 45:
                                self.image[x, y, 0] = 0
                                self.image[x, y, 1] = 0
                                self.image[x, y, 2] = 0


                    if croppedH > croppedW:
                        if cnt == 1:
                            output1 = self.image[0:self.image.shape[0] - 70, 0: self.image.shape[1]]
                        if cnt == 2:
                            output2 = self.image[70:self.image.shape[0], 0: self.image.shape[1]]

        return output1,output2



    def find_defects (self, sample):
        self.the_list = []
        gray = cv2.cvtColor(sample,
                            cv2.COLOR_BGR2GRAY)

        mask = cv2.inRange(gray, 4, 43)

        bg = cv2.bitwise_and(gray, gray, mask=mask)


        thresh = cv2.threshold(bg, 4, 255, cv2.THRESH_BINARY)[1] #remove non-defect parts

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


        cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points.
        # For example, an up-right rectangular contour is encoded with 4 points.
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        for c in cntrs:
            area = cv2.contourArea(c)
            area_thresh = 70
            if area > area_thresh:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(sample, (x, y), (x + w , y + h), (255, 0, 0), 2)
                self.the_list.append(area)
                #cv2.drawContours(input_image, [box], 0, (255, 0, 0), 2)

        return sample, self.the_list

    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()


