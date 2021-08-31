from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import dlib
import cv2

def sound_alarm(path):
    playsound.playsound(path)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A+B)/(2.0*C)
    return ear

EYE_AR_THRESH = 0.29
EYE_AR_CONSEC_FRAMES = 30

COUNTER = 0
ALARM_ON = False

print("\n\tLoading DLib Facial Landmark Detector")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

lStart =42
lEnd = 48
rStart = 36
rEnd = 42

print("\n\tStarting WebCam")
cap = cv2.VideoCapture(0)

while True:
    isTrue, frame = cap.read()
    rects = detector(frame, 0)
    for rect in rects:
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull] , -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    t = Thread(target=sound_alarm,args=("alarm.wav",))
                    t.deamon = True
                    t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            ALARM_ON = False

        if ear>=0.29:  
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (310, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (310, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detection - Team Aneerban, Sakshi, Abhishek", frame)

    if cv2.waitKey(1) & 0xFF == ord("Q"):
        print("\n\tThankyou for using our software!!\n")
        exit()

cv2.destroyAllWindows()