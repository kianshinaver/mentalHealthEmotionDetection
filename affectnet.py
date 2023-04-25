import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained emotion recognition model (e.g., VGG19, ResNet, etc.)
emotion_model_path = '_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path)

# Emotion labels
emotion_labels = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"]
emotion_counts = {"anger": 0, "disgust": 0, "fear": 0, "happy": 0, "neutral": 0, "sad": 0, "surprise": 0}

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces and emotions
def detect_faces_and_emotions(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = gray_frame[y:y + h, x:x + w]
        face = cv2.resize(face, (64, 64))
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face /= 255

        emotion_prediction = emotion_classifier.predict(face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]

        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        # Counting emotions
        emotion_counts[emotion_label] += 1
    return frame

# # Process the video file
# video_path = 'test.mp4'
# cap = cv2.VideoCapture(video_path)

# Process live video
cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    processed_frame = detect_faces_and_emotions(frame)
    cv2.imshow('Video', processed_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.bar(emotion_counts.keys(), emotion_counts.values())
plt.title("Emotion Quantities")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.show()
