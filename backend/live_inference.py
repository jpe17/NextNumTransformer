"""
Live Vision Transformer Inference
=================================
Real-time digit recognition - just hold up your numbers!
"""

import torch
import cv2
import numpy as np
from utils import load_trained_model, predict_sequence

def process_region(region):
    """Convert camera region to model input - white bg + black digits."""
    # Convert to grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    
    # Resize to canvas dimensions (30x120)
    canvas = cv2.resize(gray, (120, 30), interpolation=cv2.INTER_AREA)
    
    # Light blur to reduce noise
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    
    # Since you trained on white bg + black digits, keep it as-is
    # Just normalize to [0, 1] where 0=black digits, 1=white background
    canvas_tensor = torch.from_numpy(canvas.astype(np.float32)) / 255.0
    
    # Create patches
    patch_size = 6
    patches = canvas_tensor.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    patches = patches.contiguous().view(-1, patch_size, patch_size)
    patches = patches.unsqueeze(1)  # [100, 1, 6, 6]
    
    return patches

def main():
    """Real-time inference with live camera."""
    print("🚀 Starting Live Digit Recognition!")
    print("📝 Hold up handwritten digits (black on white paper)")
    print("🟢 Position digits inside the green box")
    print("⚡ Predictions update in real-time!")
    print("🚪 Press 'q' to quit\n")
    
    # Load model using modular function
    try:
        model, config = load_trained_model(run_folder='run_25_06_25__1_54_21')  # Use latest model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Start camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Get camera dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Box dimensions (1:4 ratio)
    box_height = height // 4
    box_width = box_height * 4
    box_x = (width - box_width) // 2
    box_y = (height - box_height) // 2
    
    print(f"📹 Camera ready: {width}x{height}")
    print(f"📦 Capture box: {box_width}x{box_height}")
    
    # Prediction tracking
    last_prediction = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract region inside box
        region = frame[box_y:box_y+box_height, box_x:box_x+box_width]
        
        # Run prediction every few frames (for performance)
        if frame_count % 3 == 0:  # Every 3rd frame
            try:
                patches = process_region(region)
                # Use modular prediction function
                prediction = predict_sequence(model, patches, config.get('max_seq_len', 6))
                if prediction:  # Only update if we got a prediction
                    last_prediction = prediction
            except:
                pass  # Skip frame if processing fails
        
        frame_count += 1
        
        # Draw green box
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                     (0, 255, 0), 3)
        
        # Show prediction in large text
        pred_text = ''.join(map(str, last_prediction)) if last_prediction else "---"
        cv2.putText(frame, f"Prediction: {pred_text}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # Instructions
        cv2.putText(frame, "Hold digits in green box", (20, height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (20, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Live Digit Recognition', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Goodbye!")

if __name__ == "__main__":
    main() 