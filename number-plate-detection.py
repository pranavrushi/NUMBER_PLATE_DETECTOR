#import lib

import cv2
import imutils as im

input = 'car2.png'
image = cv2.imread(input)
cv2.imshow("Input Image", image)
new_width = 500
image = im.resize(image, width=new_width) #resizing the image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converting bgr to gray scaling

d, sigmaColor, sigmaSpace = 11,17,17
filtered_image = cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)

lower, upper = 170, 200
edged = cv2.Canny(filtered_image, lower, upper)

# lets find the boundries of the iamge or say contours
cnts,hir = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
NumberPlateCnt = None
print("Number of Contours found : " + str(len(cnts)))

count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        epsilon = 0.01 * peri
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:  
            print(approx)
            NumberPlateCnt = approx 
            break
            
#applying detection

cv2.imshow("Gray scale Image", gray)
cv2.imshow("After Applying Bilateral Filter", filtered_image)
cv2.imshow("After Canny Edges", edged)

cv2.drawContours(image, [NumberPlateCnt], -1, (255,0,0), 2)
cv2.imshow("Output", image) #getting output image

cv2.waitKey(0) 