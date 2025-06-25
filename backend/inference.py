"""
Vision Transformer Inference
============================
Simple inference script that reuses existing code.
"""

import torch
import os
from transformer_architecture import VisionTransformer
from data_processing import get_data, visualize_canvas_sequence

# Special tokens
SOS_TOKEN = 10
EOS_TOKEN = 11
PAD_TOKEN = 12

def load_trained_model():
    """Load the trained model from artifacts."""
    # Model config (should match training)
    model_config = {
        'patch_dim': 36, 'embed_dim': 128, 'num_patches': 100,
        'num_heads': 2, 'num_layers': 5, 'ffn_ratio': 2,
        'vocab_size': 13, 'max_seq_len': 6
    }
    
    # Load model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, "artifacts", "trained_model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first!")
    
    model = VisionTransformer(**model_config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, model_config

def predict_sequence(model, canvas_patches, max_seq_len=6):
    """Autoregressive inference to predict digit sequence."""
    canvas_patches = canvas_patches.unsqueeze(0)  # Add batch dim
    generated_sequence = [SOS_TOKEN]
    
    with torch.no_grad():
        for _ in range(max_seq_len - 1):
            # Prepare decoder input with padding
            decoder_input = generated_sequence + [PAD_TOKEN] * (max_seq_len - len(generated_sequence))
            decoder_input = torch.tensor([decoder_input], dtype=torch.long)
            
            # Get predictions
            logits = model(canvas_patches, decoder_input)
            next_token_logits = logits[0, len(generated_sequence)-1]
            next_token = torch.argmax(next_token_logits).item()
            
            generated_sequence.append(next_token)
            if next_token == EOS_TOKEN:
                break
    
    # Return only digit tokens (0-9)
    return [token for token in generated_sequence[1:] if token < 10]

def infer_single_image(image_index=0):
    """Run inference on a single test image."""
    print(f"=== Inference on Test Image {image_index} ===")
    
    # Load model and test data
    model, config = load_trained_model()
    test_patches, test_labels = get_data('test', num_sequences=50, max_digits=5)
    
    if image_index >= len(test_patches):
        print(f"❌ Index {image_index} out of range. Max: {len(test_patches)-1}")
        return
    
    # Get ground truth
    true_digits = [token.item() for token in test_labels[1][image_index] if token.item() < 10]
    
    # Visualize the image
    visualize_canvas_sequence(test_patches, test_labels, sequence_idx=image_index)
    
    # Predict
    predicted_digits = predict_sequence(model, test_patches[image_index])
    
    # Show results
    print(f"Ground Truth: {true_digits}")
    print(f"Prediction:   {predicted_digits}")
    print(f"Correct: {'✅' if predicted_digits == true_digits else '❌'}")
    
    return predicted_digits, true_digits

def run_inference_demo():
    """Demo inference on multiple test images."""
    print("=== Vision Transformer Inference Demo ===\n")
    
    model, config = load_trained_model()
    test_patches, test_labels = get_data('test', num_sequences=100, max_digits=5)
    
    correct = 0
    total = 100  # Test on 5 images
    
    for i in range(total):
        true_digits = [token.item() for token in test_labels[1][i] if token.item() < 10]
        predicted_digits = predict_sequence(model, test_patches[i])
        
        print(f"Image {i}: True={true_digits}, Pred={predicted_digits}, "
              f"{'✅' if predicted_digits == true_digits else '❌'}")
        
        if predicted_digits == true_digits:
            correct += 1
    
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}")

if __name__ == "__main__":
    # Run demo
    run_inference_demo()
    
    # Or infer a specific image:
    # infer_single_image(image_index=3) 