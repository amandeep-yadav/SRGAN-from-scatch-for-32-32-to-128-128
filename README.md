# Super-Resolution Image Upscaling with SRGAN

This project implements and trains a Super-Resolution Generative Adversarial Network (SRGAN) from scratch to upscale low-resolution images into high-resolution counterparts. The pipeline includes training, evaluation, and visualization of results, making it a comprehensive framework for super-resolution tasks.

---

## 🌟 Features

- **End-to-End Training**: Build and train the SRGAN model from scratch for super-resolution tasks.
- **Image Upscaling**: Enhance image quality by converting low-resolution images (32x32) to high-resolution (128x128).
- **Performance Metrics**: Evaluate model performance using:
  - **SSIM**: Structural Similarity Index
  - **PSNR**: Peak Signal-to-Noise Ratio
- **Visualization**: Compare downscaled, predicted, and ground truth images side by side for qualitative evaluation.

---

## 📂 Project Structure

```
├── image_upscale_eval.py     # Script for model evaluation and visualization
├── test.py                   # Script for testing the model
├── requirements.txt          # Dependencies for the project
├── training/                 # Contains scripts and configurations for training the SRGAN
├── data/                     # Dataset (low-res and high-res images)
│   ├── kaggle_folder_32/     # Low-resolution images (32x32)
│   └── kaggle_folder_128/    # High-resolution images (128x128)
├── models/                   # Saved model files (e.g., SRGAN generator)
└── downloads/                # Pretrained models and training checkpoints
```

---

## 🚀 Quick Start

### 1️⃣ Prerequisites

- Install Python 3.8 or above.
- Clone this repository:

```bash
git clone https://github.com/your-repo/srgan-image-upscaling.git
cd srgan-image-upscaling
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Training the SRGAN

The SRGAN training pipeline builds and trains both the generator and discriminator networks:

1. Place the training dataset in the `data/` directory.
   - **Low-resolution images**: `data/kaggle_folder_32/`
   - **High-resolution images**: `data/kaggle_folder_128/`
2. Run the training script:

```bash
python training/srgan_train.py
```

- The trained model will be saved in the `models/` directory.

---

### 3️⃣ Model Evaluation

Use the `image_upscale_eval.py` script to evaluate the trained SRGAN model on test images:

```bash
python image_upscale_eval.py
```

- **Input Directories**:
  - Low-resolution test images: `data/kaggle_folder_32/`
  - High-resolution ground truth images: `data/kaggle_folder_128/`
- **Outputs**:
  - SSIM and PSNR scores for quantitative analysis.
  - Visualizations comparing predicted and ground truth images.

---

### 4️⃣ Testing the Model

To quickly test the model with pre-trained weights, use `test.py`:

```bash
python test.py --model_path downloads/srgan_generator.h5 --image_path path/to/your/image.jpg
```

- Replace `--image_path` with the path to your input low-resolution image.

---

## 🏗️ Architecture Diagram

Below is the architecture of the SRGAN used in this project:

![SRGAN Architecture](https://github.com/amandeep-yadav/SRGAN-from-scatch-for-32-32-to-128-128/blob/main/img/Screen_Shot_2020-07-19_at_11.13.45_AM_zsF2pa7.png)

## 📚 Use Cases

- **Satellite Image Analysis**: Enhance low-resolution satellite imagery for better interpretation.
- **Medical Imaging**: Improve the resolution of medical scans (e.g., MRIs, CT scans).
- **Historical Image Restoration**: Upscale and restore old, low-quality photographs.
- **Video Streaming**: Enhance video quality during streaming.

---

## 📊 Performance Metrics

- **SSIM (Structural Similarity Index)**: Measures the similarity between the ground truth and predicted high-resolution images.
- **PSNR (Peak Signal-to-Noise Ratio)**: Indicates the quality of reconstruction, with higher values representing better quality.

---

## 🖼️ Visual Results

### Sample Comparisons
![32x32](https://github.com/amandeep-yadav/SRGAN-from-scatch-for-32-32-to-128-128/blob/main/img/2.PNG) 
![128x128-predicted](https://github.com/amandeep-yadav/SRGAN-from-scatch-for-32-32-to-128-128/blob/main/img/3.PNG) 
![128x128-ground-truth](https://github.com/amandeep-yadav/SRGAN-from-scatch-for-32-32-to-128-128/blob/main/img/10.PNG)
![32x32](https://github.com/amandeep-yadav/SRGAN-from-scatch-for-32-32-to-128-128/blob/main/img/5.PNG) 
![128x128-predicted](https://github.com/amandeep-yadav/SRGAN-from-scatch-for-32-32-to-128-128/blob/main/img/6.PNG)
![128x128-ground-truth](https://github.com/amandeep-yadav/SRGAN-from-scatch-for-32-32-to-128-128/blob/main/img/7.PNG)


## 📜 Dependencies

- **TensorFlow**: For deep learning model development
- **NumPy**: Array manipulation and processing
- **Matplotlib**: Visualization of results
- **scikit-image**: Evaluation metrics (SSIM, PSNR)

Install them via:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contribution

Contributions are welcome! If you have suggestions for improvements or find bugs, feel free to create an issue or submit a pull request.

---

## 📧 Contact

For queries or collaborations, reach out:

- **Email**: amandeep@example.com
- **LinkedIn**: [Amandeep Yadav](https://linkedin.com/in/amandeep-yadav)

---
