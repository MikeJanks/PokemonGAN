import cv2
import numpy as np


 
img = cv2.imread('.\\kaggle-one-shot-pokemon\\pokemon-b\\1.jpg', cv2.IMREAD_UNCHANGED)

print('Original Dimensions : ',img.shape)

cv2.imshow("Original image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


scale_percent = 220 # percent of original size
width = 28
height = 28
dim = (width, height)


# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',img.shape)
 
cv2.imshow("Resized image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


width = 350
height = 350
dim = (width, height)

# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',img.shape)
 
cv2.imshow("Resized image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


