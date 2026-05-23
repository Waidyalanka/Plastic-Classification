from flask import Flask, render_template, request

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = tf.keras.models.load_model("plastic_classifier.keras")

# Class names
class_names = ['PET', 'PP', 'PS']

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return 'No file uploaded'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load image
    img = image.load_img(filepath, target_size=(224,224))

    # Convert to array
    img_array = image.img_to_array(img)

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess image
    img_array = preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)

    # Get class
    predicted_class = class_names[np.argmax(prediction)]

    # Confidence
    confidence = np.max(prediction) * 100

    return render_template(
        'index.html',
        prediction=predicted_class,
        confidence=f"{confidence:.2f}",
        image_path=filepath
    )

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)