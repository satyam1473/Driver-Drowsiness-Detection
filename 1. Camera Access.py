import cv2
import dlib
import pyttsx3
from scipy.spatial import distance

# INITIALIZING THE pyttsx3 SO THAT
# ALERT AUDIO MESSAGE CAN BE DELIVERED
engine = pyttsx3.init()

# SETTING UP OF CAMERA TO 1 YOU
# CAN EVEN CHOOSE 0 IN PLACE OF 1
cap = cv2.VideoCapture(0)  # Change the camera index if necessary

# MAIN LOOP IT WILL RUN ALL THE UNLESS
# AND UNTIL THE PROGRAM IS BEING KILLED
# BY THE USER
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
    key = cv2.waitKey(1) & 0xFF  # Use bitwise AND to get only the last 8 bits

    # Check if the 'q' key is pressed
    if key == ord("q"):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
