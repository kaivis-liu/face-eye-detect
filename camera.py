import cv2
import numpy


cap = cv2.VideoCapture("camTest.mp4")
#cap = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#将图片转化成灰度
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    i=0
    for (x,y,w,h) in faces:
        i= i+1
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame, ('Face_%02d' % i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        z=0
        for (ex,ey,ew,eh) in eyes:
            z=z+1
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(img, ('Eye_%02d' % z), (x+ex, y+ey), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
    cv2.imshow("cammera", frame)
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()