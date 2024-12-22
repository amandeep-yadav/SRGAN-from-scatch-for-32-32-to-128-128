import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Function to upscale image using the trained model
def upscale_image(image_path, model):
    img = load_img(image_path, target_size=(32, 32))
    img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)  # Normalize and batchify
    high_res = model.predict(img_array)  # Predict upscaled image
    return (high_res[0] * 255).astype(np.uint8)  # Rescale to [0, 255]

# Function to visualize the original, downscaled, and predicted images in a single row
def visualize_comparison(actual, downscaled, predicted, titles):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, [actual, downscaled, predicted], titles):
        ax.imshow(img.astype(np.uint8))
        ax.set_title(title)
        ax.axis('off')
    plt.show()

# Define pipeline to load model and use it for predictions
def prediction_pipeline(model_path, image_paths, input_dir_32, input_dir_128):
    # Load the trained model
    model = load_model(model_path)

    # Initialize lists to store SSIM and PSNR scores
    ssim_scores = []
    psnr_scores = []

    # Process each image
    for image_path in image_paths:
        try:
            # Paths for downscaled and ground truth images
            img_path_32 = os.path.join(input_dir_32, image_path)
            img_path_128 = os.path.join(input_dir_128, image_path)

            # Load the original high-resolution image (ground truth)
            original_img = load_img(img_path_128, target_size=(128, 128))
            original_img_array = img_to_array(original_img)

            # Load the downscaled image (32x32)
            downscaled_img = load_img(img_path_32, target_size=(32, 32))
            downscaled_img_array = img_to_array(downscaled_img)

            # Predict the high-resolution image using the model
            predicted_img = upscale_image(img_path_32, model)

            # Compute SSIM and PSNR
            ssim_value = ssim(
                original_img_array, 
                predicted_img, 
                channel_axis=-1,  # Specify the color channel axis
                data_range=predicted_img.max() - predicted_img.min()
            )
            psnr_value = psnr(
                original_img_array, 
                predicted_img, 
                data_range=predicted_img.max() - predicted_img.min()
            )

            # Append scores to lists
            ssim_scores.append(ssim_value)
            psnr_scores.append(psnr_value)

            # Visualize the images side-by-side
            visualize_comparison(
                actual=original_img_array,
                downscaled=downscaled_img_array,
                predicted=predicted_img,
                titles=["Original (128x128)", "Downscaled (32x32)", "Predicted (128x128)"]
            )

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    # Print average SSIM and PSNR scores
    if ssim_scores and psnr_scores:
        print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
        print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
    else:
        print("No valid SSIM or PSNR scores to compute averages.")

# Example usage
if __name__ == "__main__":
    model_path = "30_gen_model.h5"
    input_dir_32 = "/kaggle/working/kaggle_folder_32/"
    input_dir_128 = "/kaggle/working/kaggle_folder_128/"

    image_paths = [
        "original_IMG1015_resized.jpg",
        "original_IMG1016_resized.jpg",
        "original_IMG1017_resized.jpg",
        "original_IMG1018_resized.jpg",
        "original_IMG1019_resized.jpg",
        "original_IMG1020_resized.jpg",
        "original_IMG1021_resized.jpg",
        "original_IMG1022_resized.jpg",
        "original_IMG1023_resized.jpg",
        "original_IMG1024_resized.jpg"
    ]

    prediction_pipeline(model_path, image_paths, input_dir_32, input_dir_128)
