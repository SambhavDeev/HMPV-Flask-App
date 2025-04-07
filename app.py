from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
import cv2
import os

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained ResNet50 model (ensure the file is in your project folder)
model = tf.keras.models.load_model('resnet50_best_model.h5')

def preprocess_image(image_path):
    """
    Preprocess the input image:
    - Reads the image using OpenCV.
    - Converts it from BGR to RGB.
    - Resizes it to 224x224.
    - Expands dimensions and applies ResNet50-specific preprocessing.
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the file in the uploads folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Preprocess the image and make prediction
            img = preprocess_image(file_path)
            if img is None:
                return "Invalid image", 400
            preds = model.predict(img)[0]
            # Assuming class indices: 0 = 'NORMAL', 1 = 'PNEUMONIA'
            classes = ['NORMAL', 'PNEUMONIA']
            predicted_class = classes[np.argmax(preds)]
            confidence = np.max(preds) * 100

            # Pass prediction and confidence to the result page
            return render_template('result.html', prediction=predicted_class, confidence=confidence)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
