from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Configure upload and output folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = os.path.abspath(UPLOAD_FOLDER)
app.config['OUTPUT_FOLDER'] = os.path.abspath(OUTPUT_FOLDER)

# Ensure upload and output folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load the trained generator model (make sure the model path is correct)
model_path = '30_gen_model.h5'  # Path to your trained model
generator = tf.keras.models.load_model(model_path)

# Dummy image processing function using the generator model
def process_image(image_path, output_path):
    """Simulates processing an image by using the generator model."""
    img = load_img(image_path, target_size=(32, 32))  # Resize to match input size (32x32)
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Use the generator to upscale the image
    generated_img = generator.predict(img_array)

    # Convert output to image format and save
    generated_img = np.squeeze(generated_img)  # Remove batch dimension
    generated_img = (generated_img * 255.0).astype(np.uint8)  # Convert back to [0, 255]
    generated_img_pil = Image.fromarray(generated_img)
    generated_img_pil.save(output_path)

@app.route('/')
def home():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image uploads and return processed results."""
    if 'image' not in request.files:
        return render_template('index.html', prediction_text="No file part")

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', prediction_text="No selected file")

    if file:
        # Save the uploaded image
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        # Process the image
        output_filename = 'processed_' + filename
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        process_image(input_path, output_path)

        return render_template(
            'index.html',
            prediction_text="Image successfully processed!",
            original_image=filename,
            processed_image=output_filename
        )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    """Serve processed files."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
