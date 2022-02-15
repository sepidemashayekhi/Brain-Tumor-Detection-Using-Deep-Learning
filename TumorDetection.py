
import cv2
import numpy as np 


class TumorDetection():
    def preprocess(self,image):
        self.image=cv2.resize(image,(200,200))
        self.image=cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        self.image=cv2.medianBlur(self.image,3)
        return self.image
    def cannyThreshold(self,image):
        image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        median=np.median(image_gray)
        self.lower=int(max(0,(1-0.33)*median))
        self.upper=int(min(255,(1+0.33)*median))

    def detection(self,imgPre,Thresh):
        self.thresh=cv2.threshold(imgPre,Thresh,255,cv2.THRESH_BINARY)[1]
        self.threshInv=cv2.threshold(imgPre,Thresh,255,cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        closed = cv2.morphologyEx(self.thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations = 3)
        closed = cv2.dilate(closed, None, iterations = 4)
        edged = cv2.Canny(closed,self.lower,self.upper)
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.image, cnts, -1, (0, 0, 255), 2)
        return edged

image=cv2.imread('data\yes\Y960.jpg')
TumorD=TumorDetection()
imagePre=TumorD.preprocess(image)
meanStd=cv2.meanStdDev(imagePre)
Thresh=meanStd[0][0]+meanStd[1][0]
TumorD.cannyThreshold(image)
thresh=TumorD.detection(imagePre,Thresh[0])

cv2.imshow("imageo",image)
cv2.imshow("image",imagePre)
cv2.imshow("thresh",thresh)
cv2.waitKey(0)