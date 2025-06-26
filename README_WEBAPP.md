# Vision Transformer Number Recognition - Web Application

A beautiful web interface for real-time handwritten number recognition using Vision Transformers with attention visualization.

## 🌟 Features

### 1. **Real-time Scanning with Attention** 📸
- Upload images via drag & drop or file browser
- Real-time webcam capture
- Live attention visualization showing what the model focuses on
- Interactive image analysis

### 2. **Image Processing Pipeline** ⚙️
- Visual step-by-step processing pipeline
- See how raw images are transformed for the model
- Processing steps: Original → Grayscale → Resized → Thickened

### 3. **Number Prediction Display** 🔢
- Clear prediction results
- Confidence visualization through attention maps
- Support for sequences of multiple digits

## 🚀 Quick Start

### Option 1: Simple Launch
```bash
python run_webapp.py
```

### Option 2: Manual Launch
```bash
python app.py
```

Then open http://localhost:8765 in your browser.

## 📋 Requirements

Install the dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `python-multipart` - File upload support
- `pillow` - Image processing
- `aiofiles` - Async file operations
- Plus existing ML dependencies (torch, opencv-python, etc.)

## 🎯 How to Use

### Upload an Image
1. **Drag & Drop**: Simply drag an image file onto the upload area
2. **Browse**: Click the upload area to browse and select an image
3. **Webcam**: Use the webcam section to capture images in real-time

### Analyze Images
- **🔍 Analyze Image**: Shows processing steps and prediction
- **🎯 Show Attention**: Reveals what parts of the image the model focuses on

### Supported Image Formats
- PNG, JPG, JPEG
- Handwritten digits work best
- Clear images with good contrast recommended

## 🧠 Understanding the Results

### Processing Steps
1. **Original**: Your uploaded image
2. **Grayscale**: Converted to black & white
3. **Resized**: Scaled to model input size (30x96 pixels)
4. **Thickened**: Enhanced for better digit recognition

### Attention Visualization
- **Red/Yellow areas**: High attention - where the model is looking
- **Blue/Dark areas**: Low attention - ignored regions
- **Heatmap intensity**: Shows confidence level

### Predictions
- **Single digits**: 0-9 individual numbers
- **Sequences**: Multiple digits like "123" or "4567"
- **Confidence**: Shown through attention intensity

## 🔧 Technical Details

### Model Architecture
- **Vision Transformer** with encoder-decoder structure
- **Patch-based processing** (6x6 pixel patches)
- **Multi-head attention** for focus visualization
- **Autoregressive prediction** for digit sequences

### API Endpoints
- `POST /api/predict` - Basic number prediction with processing steps
- `POST /api/predict_with_attention` - Prediction with attention visualization
- `GET /api/model_info` - Model configuration and status
- `GET /api/health` - Health check

### Performance
- **Real-time inference** (~100-500ms per image)
- **Batch processing** support
- **Memory efficient** attention calculation

## 🎨 Interface Features

### Modern Design
- **Gradient backgrounds** for visual appeal
- **Card-based layout** for organized sections
- **Responsive design** works on mobile and desktop
- **Smooth animations** and hover effects

### User Experience
- **Drag & drop** file upload
- **Real-time feedback** during processing
- **Loading indicators** for long operations
- **Error handling** with clear messages

### Accessibility
- **Clear visual hierarchy**
- **Descriptive labels** and alt text
- **Keyboard navigation** support
- **Color contrast** optimized

## 🔍 Troubleshooting

### Common Issues

**Model not loading**
- Check if trained model exists in `artifacts/run_gauss/`
- Verify all dependencies are installed
- Check console output for specific error messages

**Webcam not working**
- Grant camera permissions to your browser
- Ensure no other application is using the camera
- Try refreshing the page

**Poor predictions**
- Use clear, well-lit images
- Ensure digits are clearly visible
- Try images with good contrast
- Make sure handwriting is legible

**Slow performance**
- Close other browser tabs
- Use smaller image files
- Check available system memory

### Browser Compatibility
- **Chrome/Edge**: Full support including webcam
- **Firefox**: Full support
- **Safari**: Full support (may need webcam permissions)
- **Mobile browsers**: Upload works, webcam varies

## 📊 Model Performance

The Vision Transformer achieves:
- **High accuracy** on clear handwritten digits
- **Sequence recognition** for multiple digits
- **Attention-based** explainable AI
- **Real-time processing** suitable for web applications

## 🔮 Future Enhancements

Potential improvements:
- **Real-time webcam prediction** (currently capture-based)
- **Batch processing** multiple images
- **Model comparison** between different runs
- **Custom model upload**
- **API rate limiting** and authentication
- **Mobile app** version

## 🤝 Contributing

To extend the web application:
1. Backend changes go in `app.py`
2. Frontend changes go in `frontend/index.html`
3. New features need both API endpoints and UI components
4. Test thoroughly with various image types

## 📄 License

This web application is part of the NextNumTransformer project. 