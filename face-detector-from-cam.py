import cv2

# loading default face detector xml for face detection training
training_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# loading live video from webcam
video_stream = cv2.VideoCapture(0)

# looping over video frames
while True:
	video_status, img = video_stream.read()

	# converting image to grayscale
	grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# getting face coordinate on image
	face_coordinates = training_data.detectMultiScale(grayscale_img)

	# marking faces with rectangular border
	for (x, y, w, h) in face_coordinates:
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0),2)

	# showing gray image on screen
	cv2.imshow("Showing Loaded Image", img)

	# holding screen so users can view image
	key = cv2.waitKey(1)

	# quit if "Q" key is pressed
	if key == 81 or key==113:
		break

# releasing video cam
video_stream.release()

