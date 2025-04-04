# 01_face_dataset.py
import cv2
import os
import time

# Create dataset directory if it doesn't exist
dataset_path = 'dataset'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Load Haar cascade for face detection
face_cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(face_cascade_path):
    print(f"Error: Cascade file not found at {face_cascade_path}")
    print("Please download 'haarcascade_frontalface_default.xml' and place it in the project directory.")
    exit()

face_detector = cv2.CascadeClassifier(face_cascade_path)

# Get user ID and name
try:
    face_id = input("Enter numeric user ID and press <Enter>: ")
    face_name = input("Enter user name and press <Enter>: ")
    if not face_id.isdigit():
        print("Error: ID must be a number.")
        exit()
    face_id = int(face_id)
    print("\n[INFO] Initializing face capture for", face_name, "(ID:", face_id, "). Look at the camera and wait...")
except ValueError:
    print("Invalid input.")
    exit()


# Initialize webcam
cam = cv2.VideoCapture(0) # 0 is usually the default webcam
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

count = 0
capture_interval = 0.2 # seconds between captures
last_capture_time = time.time()
max_samples = 50 # Number of samples to take

while True:
    ret, img = cam.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30) # Minimum size of face to detect
    )

    current_time = time.time()

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Capture face sample periodically if only one face is detected
        if len(faces) == 1 and current_time - last_capture_time >= capture_interval:
            count += 1
            # Save the captured face image into the datasets folder
            file_path = os.path.join(dataset_path, f"User.{face_id}.{count}.jpg")
            cv2.imwrite(file_path, gray[y:y+h, x:x+w])
            print(f"[INFO] Captured sample {count}/{max_samples} for ID {face_id}")
            last_capture_time = current_time # Reset timer after capture

            # Display the sample number being saved
            cv2.putText(img, f"Sample: {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    cv2.imshow('Capturing Faces - Press ESC to exit', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        print("\n[INFO] Exiting face capture.")
        break
    elif count >= max_samples: # Take samples and exit
         print(f"\n[INFO] Finished capturing {max_samples} samples.")
         break

# Cleanup
print("\n[INFO] Cleaning up...")
cam.release()
cv2.destroyAllWindows()