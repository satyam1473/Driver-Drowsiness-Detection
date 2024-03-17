import cv2
import dlib
import pyttsx3
import bz2
from scipy.spatial import distance


# # SETTING UP OF CAMERA TO 1 YOU
# # CAN EVEN CHOOSE 0 IN PLACE OF 1
# cap = cv2.VideoCapture(0)  # Change the camera index if necessary

# # MAIN LOOP IT WILL RUN ALL THE UNLESS
# # AND UNTIL THE PROGRAM IS BEING KILLED
# # BY THE USER
# while True:
#     ret, frame = cap.read()
    
#     # Check if frame is not empty
#     if not ret:
#         print("Error: Couldn't read frame from the camera")
#         break

#     gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     cv2.imshow("Drowsiness DETECTOR IN OPENCV2", gray_scale)
#     key = cv2.waitKey(1) & 0xFF  # Use bitwise AND to get only the last 8 bits

#     # Check if the 'q' key is pressed
#     if key == ord("q"):
#         break

# # Release the camera and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

# Specify the path to your .bz2 file
bz2_file_path = "C:/Users/SATYAM KUMAR/Desktop/shape_predictor_68_face_landmarks.dat.bz2"
# Specify the path where you want to save the extracted .dat file
extracted_dat_file_path = "C:/Users/SATYAM KUMAR/Desktop/shape_predictor_68_face_landmarks.dat"

# Extract the contents of the .bz2 file
with bz2.open(bz2_file_path, "rb") as bz2_file:
    with open(extracted_dat_file_path, "wb") as dat_file:
        dat_file.write(bz2_file.read())

# Load the .dat file into a dlib.shape_predictor
dlib_facelandmark = dlib.shape_predictor(extracted_dat_file_path)

# Now you can use dlib_facelandmark as your shape predictor

# INITIALIZING THE pyttsx3 SO THAT
# ALERT AUDIO MESSAGE CAN BE DELIVERED
engine = pyttsx3.init()

# SETTING UP OF CAMERA TO 1 YOU CAN 
# EVEN CHOOSE 0 IN PLACE OF 1
cap = cv2.VideoCapture(0)

# FACE DETECTION OR MAPPING THE FACE 
# TO GET THE Eye AND EYES DETECTED
face_detector = dlib.get_frontal_face_detector()

# PUT THE LOCATION OF .DAT FILE (FILE 
# FOR PREDECTING THE LANDMARKS ON FACE )
#dlib_facelandmark = dlib.shape_predictor("C:/Users/SATYAM KUMAR/Desktop/shape_predictor_68_face_landmarks.dat.bz2")

# FUNCTION CALCULATING THE ASPECT RATIO 
# FOR THE Eye BY USING EUCLIDEAN DISTANCE 
# FUNCTION
def Detect_Eye(eye):
	poi_A = distance.euclidean(eye[1], eye[5])
	poi_B = distance.euclidean(eye[2], eye[4])
	poi_C = distance.euclidean(eye[0], eye[3])
	aspect_ratio_Eye = (poi_A+poi_B)/(2*poi_C)
	return aspect_ratio_Eye


# MAIN LOOP IT WILL RUN ALL THE UNLESS AND
# UNTIL THE PROGRAM IS BEING KILLED BY THE 
# USER
while True:
	null, frame = cap.read()
	#gray_scale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	gray_scale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

	faces = face_detector(gray_scale)

	for face in faces:
		face_landmarks = dlib_facelandmark(gray_scale, face)
		leftEye = [] 
		rightEye = [] 

		# THESE ARE THE POINTS ALLOCATION FOR THE
		# LEFT EYES IN .DAT FILE THAT ARE FROM 42 TO 47
		for n in range(42, 48):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			rightEye.append((x, y))
			next_point = n+1
			if n == 47:
				next_point = 42
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

		# THESE ARE THE POINTS ALLOCATION FOR THE 
		# RIGHT EYES IN .DAT FILE THAT ARE FROM 36 TO 41
		for n in range(36, 42):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			leftEye.append((x, y))
			next_point = n+1
			if n == 41:
				next_point = 36
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

	cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
	key = cv2.waitKey(9)
	if key == ord("q"):
		break
cap.release()
cv2.destroyAllWindows()
