import cv2
import base64
import io
import sys
import os
from flask import Flask, render_template, Response, request, send_from_directory
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt

# Backend script imports are now absolute from the project root
from backend.utils import load_trained_model, predict_sequence
from backend.inference import _preprocess_image
from backend.attention_visualization import get_attention_heatmap_and_prediction

# --- Initialization ---
print("🤖 Loading model...")
# Model path is now relative to the backend directory
MODEL_RUN_FOLDER = "run_position" 
model, model_config = load_trained_model(run_folder=MODEL_RUN_FOLDER, verbose=True)
print("✅ Model loaded.")

print("📷 Initializing camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
print("✅ Camera initialized.")

# Configure Flask with new template and static folder paths
app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')

# --- Helper Functions ---

def get_crop_coords(frame, model_config):
    """Calculates the coordinates for the centered crop box."""
    aspect_ratio = model_config['canvas_width'] / model_config['canvas_height']
    frame_h, frame_w, _ = frame.shape
    
    crop_w = int(frame_w * 0.8)
    crop_h = int(crop_w / aspect_ratio)
    
    if crop_h > frame_h * 0.95:
        crop_h = int(frame_h * 0.95)
        crop_w = int(crop_h * aspect_ratio)

    x1 = (frame_w - crop_w) // 2
    y1 = (frame_h - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    return x1, y1, x2, y2

def plot_steps_to_base64(steps, prediction):
    """Plots preprocessing steps and returns a base64 encoded string."""
    num_steps = len(steps)
    fig, axes = plt.subplots(1, num_steps, figsize=(2.5 * num_steps, 2.5))
    fig.suptitle(f"Prediction: {prediction}", fontsize=14, y=1.0)
    
    for i, (title, image) in enumerate(steps.items()):
        axes[i].imshow(image, cmap='gray' if "Original" not in title else None)
        axes[i].set_title(f"{i+1}. {title}", fontsize=9)
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- Flask Routes ---

@app.route('/')
def index():
    """Render the main page."""
    response = app.make_response(render_template('index.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

def gen_frames():
    """Generator for video streaming with throttled attention analysis."""
    frame_counter = 0
    last_heatmap = None
    last_prediction = None
    # Update attention every N frames. Higher number = better performance.
    ATTENTION_UPDATE_RATE = 5 

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # For performance, resize the frame ONCE at the start.
        h, w, _ = frame.shape
        target_w = 640
        target_h = int(target_w * h / w)
        display_frame = cv2.resize(frame, (target_w, target_h))

        crop_coords = get_crop_coords(display_frame, model_config)
        x1, y1, x2, y2 = crop_coords
        
        # --- Throttled Attention ---
        frame_counter += 1
        if frame_counter % ATTENTION_UPDATE_RATE == 0:
            # Time to run the expensive analysis
            cropped_for_model = display_frame[y1:y2, x1:x2]
            heatmap, prediction = get_attention_heatmap_and_prediction(cropped_for_model, model, model_config)
            if heatmap is not None:
                last_heatmap = heatmap
                last_prediction = prediction

        # Apply the most recent heatmap on EVERY frame for a smooth look
        if last_heatmap is not None:
            cropped_display = display_frame[y1:y2, x1:x2]
            display_frame[y1:y2, x1:x2] = cv2.addWeighted(cropped_display, 0.6, last_heatmap, 0.4, 0)
            if last_prediction:
                prediction_str = ''.join(map(str, last_prediction))
                cv2.putText(display_frame, f'Live: {prediction_str}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (22, 115, 249), 2)
        
        # --- Overlays ---
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (22, 115, 249), 2)

        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    response = Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/capture')
def capture():
    """Capture frame, run inference, and return results."""
    # This reads a new, raw frame directly from the camera, ensuring it does
    # not have the attention overlay or guide box from the video stream.
    success, frame = cap.read()
    if not success:
        return {"error": "Failed to capture frame"}

    # Crop a clean frame (no overlay)
    x1, y1, x2, y2 = get_crop_coords(frame, model_config)
    cropped_frame = frame[y1:y2, x1:x2]
    
    # Save and reload like inference.py does for consistent processing
    temp_path = "temp_capture.png"
    cv2.imwrite(temp_path, cropped_frame)
    
    # Get prediction and visualization steps (same as inference.py)
    patches, steps = _preprocess_image(temp_path, model_config, kernel_size=(2, 2))
    predicted_digits = predict_sequence(model, patches, model_config) if patches is not None else []
    prediction_str = ''.join(map(str, predicted_digits)) if predicted_digits else "N/A"
    
    # Generate plot
    steps_img_b64 = plot_steps_to_base64(steps, prediction_str)
    
    # Clean up temporary file
    try:
        os.remove(temp_path)
    except:
        pass
    
    return {
        "prediction": prediction_str,
        "stepsImage": steps_img_b64
    }

@app.route('/upload', methods=['POST'])
def upload():
    """Handle image uploads, run inference, and return results."""
    if 'file' not in request.files:
        return {"error": "No file part in the request"}, 400
    file = request.files['file']
    if file.filename == '':
        return {"error": "No file selected for uploading"}, 400

    if file:
        try:
            # Save uploaded file temporarily to match inference.py processing
            temp_path = "temp_upload.png"
            pil_image = Image.open(file.stream)
            pil_image.save(temp_path)
            
            patches, steps = _preprocess_image(temp_path, model_config, kernel_size=(2, 2))
            predicted_digits = predict_sequence(model, patches, model_config) if patches is not None else []
            prediction_str = ''.join(map(str, predicted_digits)) if predicted_digits else "N/A"
            
            steps_img_b64 = plot_steps_to_base64(steps, prediction_str)
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
            
            return {
                "prediction": prediction_str,
                "stepsImage": steps_img_b64
            }
        except Exception as e:
            print(f"Error processing uploaded file: {e}")
            return {"error": "Failed to process the uploaded image."}, 500
            
    return {"error": "File upload failed for an unknown reason."}, 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)