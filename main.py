
# IMPORT LIBRARIES
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model


# DATASET PATH

dataset_path = "Face"   # your dataset folder
img_size = 160


# LOAD DATASET

X = []
y = []

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (img_size, img_size))
        X.append(img)
        y.append(folder)

X = np.array(X)
y = np.array(y)

print("Total images loaded:", len(X))


# ENCODE LABELS

label_enc = LabelEncoder()
y = label_enc.fit_transform(y)

# FEATURE EXTRACTION (MobileNetV2)

base_model = MobileNetV2(weights="imagenet", include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

X = preprocess_input(X)
features = model.predict(X, batch_size=32, verbose=0)


# NORMALIZE FEATURES

scaler = StandardScaler()
features = scaler.fit_transform(features)


# TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, stratify=y, random_state=42
)


# TRAIN SVM

svm = SVC(kernel='rbf', C=50, gamma='scale', probability=True)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Model Accuracy:", acc)


# REAL-TIME FACE RECOGNITION

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y_, w, h) in faces:
        face = frame[y_:y_+h, x:x+w]

        try:
            # Preprocess
            face_resized = cv2.resize(face, (img_size, img_size))
            face_array = np.expand_dims(face_resized, axis=0)
            face_array = preprocess_input(face_array)

            # Feature extraction
            features = model.predict(face_array, verbose=0)
            features = scaler.transform(features)

            # Prediction
            prob = svm.predict_proba(features)
            confidence = np.max(prob)

            pred = svm.predict(features)
            name = label_enc.inverse_transform(pred)[0]

            #Unknown detection
            if confidence < 0.6:
                name = "Unknown"

        except:
            name = "Unknown"
            confidence = 0

        # Draw rectangle
        cv2.rectangle(frame, (x, y_), (x+w, y_+h), (0, 255, 0), 2)

        # Show name + confidence
        cv2.putText(frame, f"{name} ({confidence:.2f})",
                    (x, y_-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
