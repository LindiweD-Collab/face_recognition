# 03_face_recognition.py
import cv2
import numpy as np
import os

# Paths
cascade_path = 'haarcascade_frontalface_default.xml'
trainer_file = 'trainer/trainer.yml'

# Check if necessary files exist
if not os.path.exists(cascade_path):
    print(f"Error: Cascade file not found at {cascade_path}")
    exit()
if not os.path.exists(trainer_file):
    print(f"Error: Trainer file not found at {trainer_file}")
    print("Please run '02_face_training.py' first.")
    exit()

# Load the trained recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_file)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cascade_path)

# Define font for text on screen
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize user names dictionary/list based on training IDs
# IMPORTANT: IDs must match those used during dataset creation!
# Add names corresponding to the IDs you used in Step 1
names = {
    1: "Lindiwe Dlomo",  
    2: "Another Person", 
    # Add more names as needed
}

# Initialize webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

print("\n[INFO] Starting face recognition. Press ESC to exit.")

while True:
    ret, img = cam.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        # Recognize the face
        id_num, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        confidence_threshold = 100 
        display_name = "Unknown"
        name_color = (0, 0, 255) 

        if confidence < confidence_threshold:
            # Check if the predicted ID exists in our names map
            if id_num in names:
                display_name = names[id_num]
                name_color = (0, 255, 0) 
                confidence_text = f"{round(100 - (confidence / confidence_threshold * 100))}%" # Simple % mapping (lower raw -> higher %)
            else:
                display_name = f"ID {id_num} (Unkn)" 
                confidence_text = f"{round(confidence)}"
                name_color = (0, 255, 255) 

            
            cv2.rectangle(img, (x, y), (x+w, y+h), name_color, 2)
           
            cv2.putText(img, display_name, (x+5, y-5), font, 1, name_color, 2)
            cv2.putText(img, confidence_text, (x+5, y+h-5), font, 0.6, name_color, 1)

        else:
            
            cv2.rectangle(img, (x, y), (x+w, y+h), name_color, 2)
            
            cv2.putText(img, display_name, (x+5, y-5), font, 1, name_color, 2)


    cv2.imshow('Face Recognition - Press ESC to Exit', img)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break


print("\n[INFO] Exiting Program and cleaning up stuff.")
cam.release()
cv2.destroyAllWindows()