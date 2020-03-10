import ray
import face_recognition
import requests
from cv2 import cv2
import numpy as np
import pickle5 as pickle
import pyttsx3 
import threading
from gtts import gTTS
import os     #will be on the top
import speech_recognition as sr
from pygame import mixer
import timeit
import time

face_cascade = cv2.CascadeClassifier('/Users/JakeODonnell/Desktop/Projekt/Learning-python/faceLiveDetection/haar/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/JakeODonnell/Desktop/Projekt/Learning-python/faceLiveDetection/haar/haarcascade_eye.xml')
palm_cascade = cv2.CascadeClassifier('/Users/JakeODonnell/Desktop/Projekt/Learning-python/faceLiveDetection/haar/palm.xml')
closed_frontal_palm = cv2.CascadeClassifier('/Users/JakeODonnell/Desktop/Projekt/Learning-python/faceLiveDetection/haar/closed_frontal_palm.xml')
fist = cv2.CascadeClassifier('/Users/JakeODonnell/Desktop/Projekt/Learning-python/faceLiveDetection/haar/aGest.xml')

video_capture = cv2.VideoCapture(0)
#Start counting the blicktime
def started_blink(start2):
    print("START: " + str(start2))
    end = time.time()
    print(end - start2)
    if start2 != 0:
        if end - start2 > 1.5:
                start = time.time()
                runImageRec(known_face_encodings, known_face_names, start)
    
#process face
def runImageRec(known_face_encodings, known_face_names, start):
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    print(len(known_face_names))

    with open('dataset_faces.dat', 'rb') as f:
        try:
            all_face_encodings = pickle.load(f)
            known_face_names = list(all_face_encodings.keys())
            known_face_encodings = np.array(list(all_face_encodings.values()))

        except EOFError:
            pass

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        frame = cv2.flip(frame,1)

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            end = time.time()
            if (end - start) > 3:
                print((end - start))
                look_for_eye()
            print((end - start))

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                try:
                    best_match_index = np.argmin(face_distances)
                except ValueError:
                    print(known_face_names)
                    print("in error name")
                    pass
                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face

                try:
                    if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                    else:
                        if video_capture.isOpened:
                            name = "unknown"
                            print("Enter name")

                        if name == "asking you":
                            #name = input()
                            image_path = r'/Users/JakeODonnell/Desktop/Projekt/Learning-python/faceLiveDetection/img/known/' + name + '.PNG'
                            directory = r'/Users/JakeODonnell/Desktop/Projekt/Learning-python/faceLiveDetection/img/known'
                            filename = name + '.PNG'
                            cv2.imwrite(image_path, frame)
                            all_face_encodings = {}
                            new_image = face_recognition.load_image_file("/Users/JakeODonnell/Desktop/Projekt/Learning-python/faceLiveDetection/img/known/" + filename)
                            all_face_encodings[name] = face_recognition.face_encodings(new_image)[0]
                            with open('dataset_faces.dat', 'wb') as f:
                                pickle.dump(all_face_encodings, f)
                            try: 
                                os.remove(image_path)
                            except: pass
                    
                except UnboundLocalError:
                    print("in error")
                    if video_capture.isOpened:
                        name = "asking you"
                        print("Enter name")

                    if name == "sparr":
                        image_path = r'/Users/JakeODonnell/Desktop/Projekt/Learning-python/faceLiveDetection/img/known/' + name + '.PNG'
                        directory = r'/Users/JakeODonnell/Desktop/Projekt/Learning-python/faceLiveDetection/img/known'
                        filename = name + '.PNG'
                        cv2.imwrite(image_path, frame)
                        all_face_encodings = {}
                        new_image = face_recognition.load_image_file("/Users/JakeODonnell/Desktop/Projekt/Learning-python/faceLiveDetection/img/known/" + filename)
                        all_face_encodings[name] = face_recognition.face_encodings(new_image)[0]

                        with open('dataset_faces.dat', 'wb') as f:
                            pickle.dump(all_face_encodings, f)

                        try: 
                            os.remove(image_path)
                        except: pass
                
                face_names.append(name)
                

        # DisplayFace processed
        for (top, right, bottom, left), name in zip(face_locations, known_face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            # Only process every other frame of video to save time

            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            roi_gray = gray[right:right+top, left:left+bottom]
            roi_color = frame[right:right+top, left:left+bottom]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 2, 5)

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            

        # Display the resulting image
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('a'):
            try:
                user_to_delete = str(input())
                del all_face_encodings[user_to_delete]
            except KeyError:
                print("{user} doesn't exist in database".format(user=user_to_delete))
            break


#process eyes
def look_for_eye():
    while True:    
        ret, frame = video_capture.read()
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 2, 5)
        '''
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
        print(len(faces))
        '''
        '''
        palms,rejectLevels, levelWeights = fist.detectMultiScale3(
        frame,
        scaleFactor=1.1,
        minNeighbors=20,
        minSize=(24, 24),
        maxSize=(96,96),
        flags = cv2.CASCADE_SCALE_IMAGE,
        outputRejectLevels = True
        )

        palms = fist.detectMultiScale(gray, 2, 5)
        i = 0
        font = cv2.FONT_ITALIC
        for (x,y,w,h) in palms:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(image,str(i)+str(":")+str(np.log(levelWeights[i][0])),(x,y), font,0.5,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(frame,str("palm"),(x,y), font,0.5,(255,255,255),2,cv2.LINE_AA)
            i = i+1
        '''

        eyes,rejectLevels, levelWeights = eye_cascade.detectMultiScale3(
        frame,
        scaleFactor = 1.1,
        minNeighbors = 20,
        minSize = (24, 24),
        maxSize = (96,96),
        flags = cv2.CASCADE_SCALE_IMAGE,
        outputRejectLevels = True
        )

        i = 0
        font = cv2.FONT_ITALIC
        for (x,y,w,h) in eyes:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(image,str(i)+str(":")+str(np.log(levelWeights[i][0])),(x,y), font,0.5,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(frame,str("Eye"),(x,y), font,0.5,(255,255,255),2,cv2.LINE_AA)
            i = i+1

        if len(eyes) == 2:
            print("reset")
            running = False
            start2 = 0

        if len(eyes) == 0:
            running = True
            start2 = 0

        if len(eyes)== 1:
            try:
                if running == False:
                    start2 = time.time()
                running = True
                started_blink(start2)
            except UnboundLocalError:
                pass

        cv2.imshow('Video',frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
'''                        
video_capture = cv2.VideoCapture(0)

all_face_encodings = {}

test_image = face_recognition.load_image_file("/Users/JakeODonnell/Desktop/Projekt/Learning-python/jake.PNG")
all_face_encodings["test"] = face_recognition.face_encodings(test_image)[0]

with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)

'''
file_exists = os.path.isfile("/Users/JakeODonnell/Desktop/Projekt/Learning-python/dataset_faces.dat") 
if file_exists:
    pass
else:
   open('dataset_faces.dat','w+')

with open('dataset_faces.dat', 'rb') as f:
    try:
        all_face_encodings = pickle.load(f)
        known_face_names = list(all_face_encodings.keys())
        known_face_encodings = np.array(list(all_face_encodings.values()))
        look_for_eye()

    except EOFError:
        known_face_names = list()
        known_face_encodings = list()
        look_for_eye()

video_capture.release()
cv2.destroyAllWindows()
