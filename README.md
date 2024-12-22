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
├── requirements.txt          # Dependencies for the project
├── training/                 # Contains scripts and configurations for training the SRGAN
├── data/                     # Dataset (low-res and high-res images)
│   ├── kaggle_folder_32/     # Low-resolution images (32x32)
│   └── kaggle_folder_128/    # High-resolution images (128x128)
└── models/                   # Saved model files (e.g., SRGAN generator)
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

## 📊 Performance Metrics

- **SSIM (Structural Similarity Index)**: Measures the similarity between the ground truth and predicted high-resolution images.
- **PSNR (Peak Signal-to-Noise Ratio)**: Indicates the quality of reconstruction, with higher values representing better quality.

---

## 🖼️ Visual Results

Sample comparison of images:

| Downscaled (32x32) | Predicted (128x128) | Ground Truth (128x128) |
|--------------------|---------------------|------------------------|
| ![32x32](path-to-example) | ![128x128-predicted](path-to-example) | ![128x128-ground-truth](path-to-example) |

---

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

### ⚡ Acknowledgments

This project is inspired by the original SRGAN paper: [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

