from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('license_plate_model.h5')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def predict_license_plate(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    return predicted_class

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        result = predict_license_plate(file_path)
        return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
