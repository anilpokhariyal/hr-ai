from flask import Flask, request, redirect
from flask import render_template
import face_recognition
import cv2
import numpy as np
import os, json
from os import path
import pickle

app = Flask(__name__)


# feeding known faces data so app can compare with live cam
def known_faces():
    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []
    if not path.exists('encoding_emp.dat'):
        # Loading sample pictures to recognize employee.
        for emp_pic in os.listdir('employee-pics'):
            input_image = face_recognition.load_image_file('employee-pics/' + emp_pic)
            encoded_img = face_recognition.face_encodings(input_image)
            if encoded_img:
                known_face_encodings.append(encoded_img[0])
                known_face_names.append(emp_pic)
        # storing encoded list in a single file
        with open('encoding_emp.dat', 'wb') as face_encode_file:
            pickle.dump(known_face_encodings, face_encode_file)
        with open('emp_names.dat', 'wb') as emp_name_file:
            pickle.dump(known_face_names, emp_name_file)
    else:
        with open('encoding_emp.dat', 'rb') as face_encodes:
            known_face_encodings = pickle.load(face_encodes)
            known_face_encodings = np.array(list(known_face_encodings))
        with open('emp_names.dat', 'rb') as face_names:
            known_face_names = pickle.load(face_names)
            known_face_names = np.array(list(known_face_names))

    return known_face_encodings, known_face_names


# calling known faces and storing data in variables
known_face_encodings, known_face_names = known_faces()


def recognize_cam_face():
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    # Initialize live face recognition variables
    face_locations = []
    face_encodings = []
    face_names = []
    max_attemp_limit = 10
    match_tried = 0
    while match_tried <= max_attemp_limit:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # if not face detected in cam, still process so we can terminate the loop
        if len(face_encodings) == 0:
            match_tried = match_tried + 1

        # loop through face encodings to match with stored encodings
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            # get the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                face_names.append(known_face_names[best_match_index])

            if len(face_names) > 5 or match_tried == max_attemp_limit:
                video_capture.release()
                cv2.destroyAllWindows()
                return face_names

            match_tried = match_tried + 1

        # Display the resulting image
        # cv2.imshow('Video', frame)

    return face_names


@app.route('/')
def index():
    return render_template('signin.html', activity=False)


@app.route('/activity-ai')
def activity_ai():
    matches = recognize_cam_face()
    from_url = request.args.get('from', None)
    emp_list = []
    emp_id = 0
    for match in matches:
        emp_list.append(match.split('_')[0])
    if len(emp_list):
        emp_id = max(emp_list, key=emp_list.count)

    if from_url:
        return redirect(from_url + '?emp_id=' + str(emp_id))

    return render_template('signin.html', activity=True, emp_id=emp_id)


###############################################################################
# Entry point
###############################################################################

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
