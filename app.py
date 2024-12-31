from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
from werkzeug.utils import secure_filename

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

# Dummy image processing function (e.g., upscaling)
def process_image(image_path, output_path):
    """Simulates processing an image by resizing it."""
    img = Image.open(image_path).convert('RGB')  # Open image
    width, height = img.size
    # Resize image (double dimensions)
    processed_img = img.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
    processed_img.save(output_path)  # Save processed image

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
