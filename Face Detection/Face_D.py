import face_recognition
import numpy as np


n = 15
known_face_encodings = []

def load_image(img_name):
    known_face_encodings.append(np.array(face_recognition.face_encodings(face_recognition.load_image_file(img_name))[0]))


for i in range(1,n+1):
    load_image('data/'+str(i)+'.jpg')


unknown_image = face_recognition.load_image_file("data/test1.jpg")
unknown_face_encodings = np.array(face_recognition.face_encodings(unknown_image))

#print(unknown_face_encodings)
for unknown_face_encoding in unknown_face_encodings:
    results = face_recognition.compare_faces(known_face_encodings, unknown_face_encodings, tolerance=0.7)
    name = "Unknown"

    for i in range(n):
        if results[i]:
            name = "Person" + str(i+1)
    print(f"Found {name} in the photo!")
   

