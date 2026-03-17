from flask import Flask, request, render_template, redirect
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load emotion model
model = load_model('emotion_detection_model.hdf5', compile=False)

# Emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':

        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        image_path = 'static/uploaded_image.jpg'
        file.save(image_path)

        # Read image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        predicted_emotion = "No face detected"
        confidence = 0

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (64, 64))
            face = face / 255.0
            face = np.reshape(face, (1, 64, 64, 1))

            prediction = model.predict(face, verbose=0)

            emotion_index = np.argmax(prediction)
            predicted_emotion = emotions[emotion_index]
            confidence = round(prediction[0][emotion_index] * 100, 2)

            break

        return render_template(
            'index.html',
            image_path=image_path,
            emotion=predicted_emotion,
            confidence=confidence
        )

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)