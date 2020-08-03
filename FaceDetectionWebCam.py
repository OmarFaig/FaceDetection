import cv2
from random import randrange

trained_face_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


webcam=cv2.VideoCapture(0)
# in folgende While Schleife wird ständig(nahezu Echtzeit) die Aufnahme erkannt und in schwarz-weiss umgewandelt. Mit dem Taster "Q" wird der Programm abgebrochen
while True :
        successful_frame_read,frame=webcam.read()
        grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (randrange(1, 256), randrange(1, 256), randrange(1, 256)), 5)

        cv2.imshow("FACE detection", frame)
        key = cv2.waitKey(1)

        if key ==81 or key ==113:
            break

print ("Code compeleted")
