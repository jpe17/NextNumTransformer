"""
Vision Transformer Inference
============================
Simple inference script that reuses existing code.
"""

import torch
from data_processing import get_data, visualize_canvas_sequence
from utils import load_config, load_trained_model, predict_sequence

config = load_config('backend/config.json')

# Special tokens (keeping for backward compatibility, but they're imported in utils)
SOS_TOKEN = 10
EOS_TOKEN = 11
PAD_TOKEN = 12

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
    predicted_digits = predict_sequence(model, test_patches[image_index], config.get('max_seq_len', 6))
    
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
        predicted_digits = predict_sequence(model, test_patches[i], config.get('max_seq_len', 6))
        
        print(f"Image {i}: True={true_digits}, Pred={predicted_digits}, "
              f"{'✅' if predicted_digits == true_digits else '❌'}")
        
        if predicted_digits == true_digits:
            correct += 1
    
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}")

if __name__ == "__main__":
    # Run demo
    run_inference_demo(run_folder="run_25_06_25__1_54_21")
    
    # Or infer a specific image:
    # infer_single_image(image_index=3, run_folder="run_25_06_25__1_54_21") 