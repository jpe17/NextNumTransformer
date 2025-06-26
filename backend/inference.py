"""
Vision Transformer Inference
============================
Simple inference script for webcam and demo testing.
"""

import torch
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import load_trained_model, predict_sequence
from attention_visualization import create_real_time_attention_overlay

# Special tokens (keeping for backward compatibility, but they're imported in utils)
SOS_TOKEN = 10
EOS_TOKEN = 11
PAD_TOKEN = 12

def _preprocess_image(image_path, model_config, kernel_size=(2, 2)):
    """
    Loads and preprocesses a REAL-WORLD image (camera/file) for the model.
    
    NOTE: This function is ONLY used for real-world images (webcam, uploads, etc.)
    and NOT for synthetic test data, which is already in the correct format.
    
    Returns patches and intermediate steps for visualization.
    """
    try:
        original_image = Image.open(image_path)
    except FileNotFoundError:
        print(f"❌ Image not found at {image_path}"); return None, None

    # Get dimensions from model config
    canvas_height = model_config['canvas_height']
    canvas_width = model_config['canvas_width']
    patch_size = model_config['patch_size']
    
    grayscale_image = transforms.Grayscale()(original_image)
    resized_image = transforms.Resize((canvas_height, canvas_width))(grayscale_image)
    resized_image = np.clip(((np.array(resized_image, dtype=np.float32) / 255 -0.45)*8)+0.45, 0, 1)
    
    # Main processing pipeline
    eroded_canvas = cv2.erode(resized_image, np.ones(kernel_size, np.uint8))
    thickened_canvas = torch.from_numpy(eroded_canvas.astype(np.float32))

    # Patchify for model
    patches = thickened_canvas.unsqueeze(0).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(-1, patch_size, patch_size).unsqueeze(1)

    steps = {
        "Original": original_image, "Grayscale": grayscale_image, "Resized": resized_image, "Thickened": thickened_canvas.numpy()
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

def infer_from_webcam(run_folder=None, visualize=False, show_attention=False):
    """
    Captures an image from the webcam, saves it, and runs inference.
    A rectangle is drawn on screen to guide aspect ratio.
    
    Args:
        run_folder: Model run folder to use
        visualize: Whether to show preprocessing steps after capture
        show_attention: Whether to show LIVE real-time attention overlay on webcam feed
                       (shows what parts of the image the model is focusing on in real-time)
    """
    # Load model config first to get proper aspect ratio
    model, model_config = load_trained_model(run_folder=run_folder, verbose=False)
    
    # Calculate aspect ratio from model config
    canvas_height = model_config['canvas_height']
    canvas_width = model_config['canvas_width']
    aspect_ratio = canvas_width / canvas_height
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("\n📷 Webcam opened. Position digits inside the green rectangle.")
    if show_attention:
        print("🔥 LIVE ATTENTION MODE: You'll see red/yellow highlights showing where the model is looking!")
    print("Press 'c' to capture, 'q' to quit.")
    
    # Get frame dimensions once
    ret, frame = cap.read()
    if not ret:
        print("❌ Can't receive frame. Exiting ...")
        cap.release()
        return
    frame_h, frame_w, _ = frame.shape

    # Define the cropping box based on model's actual aspect ratio
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

        # Create display frame with rectangle and text
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, "Fit digits in this box", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add real-time attention overlay if requested
        if show_attention:
            display_frame = create_real_time_attention_overlay(
                display_frame, model, model_config, (x1, y1, x2, y2), alpha=0.4
            )

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
            
            # Run inference on the saved image using already loaded model
            # REAL-WORLD CAMERA IMAGE: Apply preprocessing to convert to model format
            patches, steps = _preprocess_image(image_path, model_config, kernel_size=(2, 2))
            if patches is not None:
                predicted_digits = predict_sequence(model, patches, model_config)
                print(f"Prediction: {predicted_digits}")
                if visualize:
                    _plot_steps(steps, predicted_digits)
            break
            
    cap.release()
    cv2.destroyAllWindows()

def run_inference_demo(run_folder=None, num_samples=10):
    """Demo inference on multiple test images."""
    print("=== Vision Transformer Inference Demo ===\n")
    
    # Load model and generate test data using model config
    model, model_config = load_trained_model(run_folder=run_folder)
    
    # Generate SYNTHETIC test data - already in correct format, no processing needed
    from data_processing import get_data
    test_patches, test_labels = get_data('test', num_sequences=num_samples, max_digits=model_config['max_digits'])
    
    correct = 0
    total = min(num_samples, len(test_patches))
    
    for i in range(total):
        true_digits = [token.item() for token in test_labels[1][i] if token.item() < 10]
        # Direct prediction on synthetic test data - NO preprocessing applied
        predicted_digits = predict_sequence(model, test_patches[i], model_config)
        
        print(f"Image {i}: True={true_digits}, Pred={predicted_digits}, "
              f"{'✅' if predicted_digits == true_digits else '❌'}")
        
        if predicted_digits == true_digits:
            correct += 1
    
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}")

if __name__ == "__main__":
    # Run demo
    run_inference_demo(run_folder="run_gauss", num_samples=100)
    
    print("\n" + "="*50)
    print("Next: Run inference from your webcam.")
    print("A window will open with your webcam feed.")
    print("Position your handwritten digits, then press 'c' to see the prediction.")
    print("="*50 + "\n")

    # 🔥 LIVE ATTENTION VISUALIZATION! 🔥
    # This will show real-time attention overlay on your webcam feed
    # Watch as the model focuses on different parts of your digits!
    infer_from_webcam(run_folder="run_gauss", visualize=True, show_attention=True)