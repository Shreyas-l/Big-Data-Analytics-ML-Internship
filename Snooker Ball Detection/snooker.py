import numpy as np
import cv2
import random

img = cv2.imread("snooker.png",1)
cv2.imshow("Original",img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (1,1),0)

thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 205, 1)
cv2.imshow("Binary",thresh)

contours,thre = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

filtered = []
for c in contours:
	approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
	area = cv2.contourArea(c)
	if ((len(approx) > 8) & (area > 400) & (area < 900 )):
		filtered.append(c)


print(len(filtered))

objects = np.zeros([img.shape[0],img.shape[1],3], 'uint8')
for c in filtered:
	col = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
	cv2.drawContours(objects,[c], -1, col, -1)
	area = cv2.contourArea(c)
	p = cv2.arcLength(c,True)
	print(area,p)

cv2.imshow("Contours",objects)
	

cv2.waitKey(0)
cv2.destroyAllWindows()
