import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load trained model
model = tf.keras.models.load_model("plastic_classifier.keras")

# Class labels
class_names = ['PET', 'PP', 'PS']

# Image path
img_path = r"C:\Users\Ishan\Desktop\plastic classification\Raw_images\validation\PP\61T-oOkvAPL.jpg"

# Load image
img = image.load_img(img_path, target_size=(224,224))

# Convert image to array
img_array = image.img_to_array(img)

# Expand dimensions
img_array_expanded = np.expand_dims(img_array, axis=0)

# Preprocess for MobileNetV2
img_array_expanded = preprocess_input(img_array_expanded)

# Predict
prediction = model.predict(img_array_expanded)

# Get class prediction
predicted_class = class_names[np.argmax(prediction)]

# Confidence score
confidence = np.max(prediction) * 100

# Display image
plt.imshow(img.astype("uint8") if hasattr(img, "astype") else np.array(img).astype("uint8"))
plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")
plt.axis("off")

# Show image window
plt.show()

# Print probabilities
print("\nClass Probabilities:")

for i, class_name in enumerate(class_names):
    print(f"{class_name}: {prediction[0][i] * 100:.2f}%")

print(f"\nFinal Prediction: {predicted_class}")