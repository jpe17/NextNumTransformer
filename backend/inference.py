"""
Vision Transformer Inference
============================
Simple inference script that reuses existing code.
"""

import torch
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from data_processing import get_data, visualize_canvas_sequence
from utils import load_config, load_trained_model, predict_sequence

config = load_config('backend/config.json')

# Special tokens (keeping for backward compatibility, but they're imported in utils)
SOS_TOKEN = 10
EOS_TOKEN = 11
PAD_TOKEN = 12

def _preprocess_image(image_path, config, kernel_size=(2, 2)):
    """Loads and preprocesses an image for the model, returning patches and intermediate steps."""
    try:
        original_image = Image.open(image_path)
    except FileNotFoundError:
        print(f"❌ Image not found at {image_path}"); return None, None

    # Config and transforms
    canvas_height, canvas_width = 30, 120
    patch_size = int(config.get('patch_dim', 36) ** 0.5)
    
    grayscale_image = transforms.Grayscale()(original_image)
    resized_image = transforms.Resize((canvas_height, canvas_width))(grayscale_image)
    
    # Main processing pipeline
    binarized_canvas = torch.where(transforms.ToTensor()(resized_image) > 0.5, 1.0, 0.0)
    eroded_canvas = cv2.erode((binarized_canvas.squeeze().numpy() * 255).astype(np.uint8), np.ones(kernel_size, np.uint8))
    thickened_canvas = torch.from_numpy(eroded_canvas.astype(np.float32) / 255.0)

    # Patchify for model
    patches = thickened_canvas.unsqueeze(0).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(-1, patch_size, patch_size).unsqueeze(1)

    steps = {
        "Original": original_image, "Grayscale": grayscale_image, "Resized": resized_image,
        "Binarized": binarized_canvas.squeeze().numpy(), "Thickened": thickened_canvas.numpy()
    }
    return patches, steps

def _plot_steps(steps, prediction):
    """Plots the preprocessing steps in a single figure."""
    fig, axes = plt.subplots(1, len(steps), figsize=(5 * len(steps), 5))
    fig.suptitle(f"Prediction: {prediction}", fontsize=20, y=1.05)
    for i, (title, image) in enumerate(steps.items()):
        axes[i].imshow(image, cmap='gray' if "Original" not in title else None)
        axes[i].set_title(f"{i+1}. {title}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def infer_from_image_path(image_path, run_folder=None):
    """Run inference on a single image from a file path."""
    print(f"=== Inference on Image from Path: {image_path} ===")
    model, config = load_trained_model(run_folder=run_folder)
    
    patches, _ = _preprocess_image(image_path, config)
    if patches is None: return

    predicted_digits = predict_sequence(model, patches, config)
    print(f"Prediction: {predicted_digits}")
    return predicted_digits

def visualize_and_predict_from_path(image_path, run_folder=None):
    """Loads an image, shows preprocessing steps, and predicts digits."""
    print(f"=== Visual Inference on Image from Path: {image_path} ===")
    model, config = load_trained_model(run_folder=run_folder, verbose=False)
    
    # Make digits thicker by using a larger kernel for erosion
    patches, steps = _preprocess_image(image_path, config, kernel_size=(2, 2))
    if patches is None: return

    predicted_digits = predict_sequence(model, patches, config)
    print(f"Prediction: {predicted_digits}")
    
    _plot_steps(steps, predicted_digits)
    return predicted_digits

def infer_from_webcam(run_folder=None, visualize=False):
    """
    Captures an image from the webcam, saves it, and runs inference.
    A rectangle is drawn on screen to guide aspect ratio.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("\n📷 Webcam opened. Position digits inside the green rectangle.")
    print("Press 'c' to capture, 'q' to quit.")
    
    # Get frame dimensions once
    ret, frame = cap.read()
    if not ret:
        print("❌ Can't receive frame. Exiting ...")
        cap.release()
        return
    frame_h, frame_w, _ = frame.shape

    # Define the cropping box based on model's aspect ratio (120/30 = 4:1)
    aspect_ratio = 120 / 30
    crop_w = int(frame_w * 0.9)  # Use 90% of the frame width for the box
    crop_h = int(crop_w / aspect_ratio)
    
    # Center the box
    x1 = (frame_w - crop_w) // 2
    y1 = (frame_h - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Can't receive frame. Exiting ...")
            break

        # Draw the rectangle and text on the frame
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, "Fit digits in this box", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Webcam - Press "c" to Capture', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Crop the original frame using the rectangle coordinates
            cropped_frame = frame[y1:y2, x1:x2]

            # Save the cropped frame
            image_path = "webcam_capture.png"
            cv2.imwrite(image_path, cropped_frame)
            print(f"✅ Cropped image saved to {image_path}")
            
            # Run inference on the saved image
            if visualize:
                visualize_and_predict_from_path(image_path, run_folder)
            else:
                infer_from_image_path(image_path, run_folder)
            break
            
    cap.release()
    cv2.destroyAllWindows()

def infer_single_image(image_index=0, run_folder=None):
    """Run inference on a single test image."""
    print(f"=== Inference on Test Image {image_index} ===")
    
    # Load model and test data using modular function
    model, config = load_trained_model(run_folder=run_folder)
    test_patches, test_labels = get_data('test', num_sequences=50, max_digits=5)
    
    if image_index >= len(test_patches):
        print(f"❌ Index {image_index} out of range. Max: {len(test_patches)-1}")
        return
    
    # Get ground truth
    true_digits = [token.item() for token in test_labels[1][image_index] if token.item() < 10]
    
    # Visualize the image
    visualize_canvas_sequence(test_patches, test_labels, sequence_idx=image_index)
    
    # Predict using modular function
    predicted_digits = predict_sequence(model, test_patches[image_index], config)
    
    # Show results
    print(f"Ground Truth: {true_digits}")
    print(f"Prediction:   {predicted_digits}")
    print(f"Correct: {'✅' if predicted_digits == true_digits else '❌'}")
    
    return predicted_digits, true_digits

def run_inference_demo(run_folder=None, num_samples=100):
    """Demo inference on multiple test images."""
    print("=== Vision Transformer Inference Demo ===\n")
    
    # Load model using modular function
    model, config = load_trained_model(run_folder=run_folder)
    test_patches, test_labels = get_data('test', num_sequences=num_samples, max_digits=5)
    
    correct = 0
    total = min(num_samples, len(test_patches))
    
    for i in range(total):
        true_digits = [token.item() for token in test_labels[1][i] if token.item() < 10]
        # Use modular prediction function
        predicted_digits = predict_sequence(model, test_patches[i], config)
        
        print(f"Image {i}: True={true_digits}, Pred={predicted_digits}, "
              f"{'✅' if predicted_digits == true_digits else '❌'}")
        
        if predicted_digits == true_digits:
            correct += 1
    
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}")

if __name__ == "__main__":
    # Run demo
    run_inference_demo(run_folder="run_25_06_25__2_13_84")
    
    print("\n" + "="*50)
    print("Next: Run inference from your webcam.")
    print("A window will open with your webcam feed.")
    print("Position your handwritten digits, then press 'c' to see the prediction and visualization.")
    print("="*50 + "\n")

    # Or capture from webcam with visualization:
    infer_from_webcam(run_folder="run_25_06_25__2_13_84", visualize=True)

    # Or infer a specific image:
    # infer_single_image(image_index=3, run_folder="run_25_06_25__1_54_21") 
    
    # Or infer from a custom image path:
    # infer_from_image_path('path/to/your/image.png', run_folder="run_25_06_25__1_54_21")
    
    # Or capture from webcam:
    # infer_from_webcam(run_folder="run_25_06_25__1_54_21")