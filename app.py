from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model(r'C:\Users\jones\OneDrive\Desktop\Brain_Tumor_Detection\Brain_tumor_model.keras')

# Define the categories
categories = ["glioma", "meningioma", "notumor", "pituitary"]

# Set the image size
image_size = (150, 150)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the file and preprocess it
        img = load_img(io.BytesIO(file.read()), target_size=image_size)  # Convert FileStorage to BytesIO and load image
        img_array = img_to_array(img)  # Convert image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Rescale pixel values

        # Perform prediction
        predictions = model.predict(img_array)
        predicted_class = categories[np.argmax(predictions)]
        confidence = np.max(predictions)

        return jsonify({
            "prediction": predicted_class,
            "Accuracy": float(confidence)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
