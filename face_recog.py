import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle


from collections import Counter
import linecache
import os
import tracemalloc


path = 'student_images'

images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)

def markAttendance(name):
    with open('attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'n{name}, {time}, {date}')


def preprocess_face(img):
    # Convert to grayscale for histogram equalization
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    equ = cv2.equalizeHist(gray)
    return equ

# take pictures from webcam 
tracemalloc.start()
cap  = cv2.VideoCapture(0)
while True:
    
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = preprocess_face(imgS)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS,number_of_times_to_upsample=2)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory usage: {current / 10**6} MB (Peak: {peak / 10**6} MB)")
    for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        print(matchIndex)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Memory usage: {current / 10**6} MB (Peak: {peak / 10**6} MB)")
        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()
            y1,x2,y2,x1 = faceloc
            # since we scaled down by 4 times
            y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
            markAttendance(name)
            
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


tracemalloc.stop()
current, peak = tracemalloc.get_traced_memory()
print(f"Memory usage: {current / 10**6} MB (Peak: {peak / 10**6} MB)")
# Stop memory tracking
tracemalloc.stop()

