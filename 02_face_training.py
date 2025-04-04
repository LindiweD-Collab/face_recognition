# 02_face_training.py
import cv2
import numpy as np
from PIL import Image # Pillow library for image handling
import os

# Path for face image database
dataset_path = 'dataset'
trainer_path = 'trainer'
model_file = os.path.join(trainer_path, 'trainer.yml')

# Create trainer directory if it doesn't exist
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

# Function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    faceSamples = []
    ids = []
    print(f"[INFO] Reading {len(imagePaths)} images from dataset...")

    for imagePath in imagePaths:
        try:
            # Open the image and convert it to grayscale
            PIL_img = Image.open(imagePath).convert('L') # L = Luminance (grayscale)
            img_numpy = np.array(PIL_img, 'uint8')

            # Get the user id from the image filename (e.g., User.1.5.jpg -> id=1)
            filename = os.path.basename(imagePath)
            if not filename.startswith("User."):
                print(f"Skipping file with unexpected format: {filename}")
                continue

            id_str = filename.split(".")[1]
            if not id_str.isdigit():
                 print(f"Skipping file with non-numeric ID: {filename}")
                 continue

            id = int(id_str)

            # Append the image numpy array and its corresponding id
            faceSamples.append(img_numpy)
            ids.append(id)
            # print(f"Processed ID: {id} from {filename}") # Uncomment for debugging

        except Exception as e:
            print(f"Error processing image {imagePath}: {e}")

    if not faceSamples or not ids:
        print("[ERROR] No faces found in the dataset directory or error reading files.")
        return None, None

    print(f"[INFO] Found {len(np.unique(ids))} unique IDs: {np.unique(ids)}")
    return faceSamples, ids

print ("\n[INFO] Training faces. This might take a bit...")

# Using LBPH (Local Binary Patterns Histograms) Recognizer
# You can also try EigenFaceRecognizer or FisherFaceRecognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces, ids = getImagesAndLabels(dataset_path)

if faces is None or ids is None:
    print("[ERROR] Could not get data for training. Exiting.")
    exit()

# Train the model using the faces and ids
try:
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write(model_file)

    # Print the number of faces trained and end program
    print(f"\n[INFO] {len(np.unique(ids))} faces trained. Model saved as {model_file}")
    print("[INFO] Training complete.")

except cv2.error as e:
     print(f"[ERROR] OpenCV error during training: {e}")
     print("This might happen if there are not enough samples or users (e.g., need at least 2 users for some recognizers).")
except Exception as e:
     print(f"[ERROR] An unexpected error occurred during training: {e}")