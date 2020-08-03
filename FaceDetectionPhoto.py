import cv2
from random import randrange

trained_face_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')





img=cv2.imread("RJD.png")
grayscaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

(x, y, w, h) = face_coordinates[0]
cv2.rectangle(img, (x, y), (x + w, y + h),
                          (randrange(1, 256), randrange(1, 256), randrange(1, 256)), 5)

cv2.imshow("FACE detection", img)
cv2.waitKey()


print ("Code compeleted")