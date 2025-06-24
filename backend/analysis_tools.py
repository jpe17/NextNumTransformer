"""
Analysis Tools Module - Clean Results Analysis
============================================

Clean visualization and analysis tools for understanding model performance.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from data_processing import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, visualize_canvas_sequence


def analyze_training_history(history, detailed=True):
    """
    Analyze training history with key insights.
    """
    print("Training Analysis:")
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    train_accs = history['train_accuracies']
    val_accs = history['val_accuracies']
    
    print(f"  Final train accuracy: {train_accs[-1]:.2f}%")
    print(f"  Final val accuracy: {val_accs[-1]:.2f}%")
    print(f"  Best val accuracy: {max(val_accs):.2f}%")
    
    # Loss reduction
    loss_reduction = ((train_losses[0] - train_losses[-1]) / train_losses[0] * 100)
    print(f"  Loss reduction: {loss_reduction:.1f}%")
    
    # Overfitting check
    overfit_gap = max(train_accs) - max(val_accs)
    if overfit_gap < 3:
        print(f"  ✓ Good generalization (gap: {overfit_gap:.1f}%)")
    else:
        print(f"  ⚠️ Some overfitting (gap: {overfit_gap:.1f}%)")
    
    if detailed:
        plot_training_curves(history)


def plot_training_curves(history):
    """
    Create clean training visualization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history['train_losses'], 'b-', label='Training', linewidth=2)
    axes[0].plot(epochs, history['val_losses'], 'r-', label='Validation', linewidth=2)
    axes[0].set_title('Loss Evolution')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, history['train_accuracies'], 'b-', label='Training', linewidth=2)
    axes[1].plot(epochs, history['val_accuracies'], 'r-', label='Validation', linewidth=2)
    axes[1].set_title('Accuracy Evolution')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Key insights
    best_epoch = np.argmax(history['val_accuracies']) + 1
    print(f"  Best epoch: {best_epoch}")
    print()


def predict_sequences(model, canvas_patches, max_len=6):
    """
    Greedy decoding for sequence prediction.
    """
    model.eval()
    batch_size = canvas_patches.size(0)
    
    # Start with SOS token
    decoder_input = torch.full((batch_size, 1), SOS_TOKEN, dtype=torch.long, device=canvas_patches.device)
    
    with torch.no_grad():
        for _ in range(max_len - 1):
            logits = model(canvas_patches, decoder_input)
            
            # Get the last token prediction
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            # Append to the sequence
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

    return decoder_input

def analyze_sequence_predictions(model, val_loader, num_batches=10):
    """
    Analyze sequence prediction performance.
    """
    print("Sequence Prediction Analysis:")
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (canvas_patches, decoder_inputs, target_outputs) in enumerate(val_loader):
            if i >= num_batches:
                break
            
            predicted_sequences = predict_sequences(model, canvas_patches)
            
            all_preds.append(predicted_sequences.cpu())
            all_labels.append(target_outputs.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # --- Metrics Calculation ---
    
    # 1. Exact Sequence Match Accuracy
    # Compare predictions to the target_outputs, ignoring padding on both
    is_pad = all_labels == PAD_TOKEN
    
    # For a prediction to be correct, all non-pad tokens must match
    # We set PAD tokens to a value that will always match to exclude them
    preds_masked = torch.where(is_pad, PAD_TOKEN, all_preds)
    
    correct_sequences = (preds_masked == all_labels).all(dim=1)
    exact_match_acc = correct_sequences.float().mean() * 100
    
    # 2. Token-level Accuracy (ignoring PAD)
    correct_tokens = (all_preds == all_labels)[~is_pad]
    token_acc = correct_tokens.float().mean() * 100

    print(f"  - Exact Sequence Match Accuracy: {exact_match_acc:.2f}%")
    print(f"  - Token-level Accuracy (non-pad): {token_acc:.2f}%")

    # 3. Per-class token accuracy
    print("  - Per-class token accuracy (non-pad):")
    for class_id in range(10): # Digits 0-9
        class_mask = (all_labels == class_id) & ~is_pad
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == all_labels[class_mask]).float().mean() * 100
            print(f"    - Class {class_id}: {class_acc:.1f}%")

    print()
    return {
        'exact_match_accuracy': exact_match_acc,
        'token_accuracy': token_acc
    }

def show_prediction_example(model, val_loader, sequence_idx=0):
    """
    Shows a single prediction example.
    """
    print("Example Prediction Analysis:")
    
    model.eval()
    
    # Get a batch from the validation loader
    try:
        canvas_patches, _, target_outputs = next(iter(val_loader))
    except StopIteration:
        print("  Could not get a batch from the validation loader.")
        return

    # Ensure the sequence_idx is valid for the batch size
    if sequence_idx >= canvas_patches.size(0):
        print(f"  sequence_idx {sequence_idx} is out of bounds for batch size {canvas_patches.size(0)}. Using index 0.")
        sequence_idx = 0

    # Select a single canvas to predict
    single_canvas_patch = canvas_patches[sequence_idx:sequence_idx+1]
    
    # Make a prediction
    predicted_sequence = predict_sequences(model, single_canvas_patch)
    
    # Visualize the canvas image
    # The second element of the tuple for labels is used for the title
    visualize_canvas_sequence(canvas_patches.cpu(), (None, target_outputs.cpu()), sequence_idx=sequence_idx)

    # Clean up predicted and true sequences for printing
    pred_labels = [p for p in predicted_sequence[0].cpu().numpy() if p not in (SOS_TOKEN, EOS_TOKEN, PAD_TOKEN)]
    true_labels = [t for t in target_outputs[sequence_idx].cpu().numpy() if t not in (EOS_TOKEN, PAD_TOKEN)]
    
    print(f"\n  - True Sequence:     {true_labels}")
    print(f"  - Predicted Sequence:  {pred_labels}")
    
    # Compare the sequences
    if true_labels == pred_labels:
        print("  - Result: ✅ Correct")
    else:
        print("  - Result: ❌ Incorrect")
    print()


def show_prediction_mistakes(model, val_loader, num_mistakes=5):
    """
    Show sequence prediction mistakes.
    """
    print("Mistake Analysis:")
    
    model.eval()
    mistakes_found = 0
    
    with torch.no_grad():
        for (canvas_patches, decoder_inputs, target_outputs) in val_loader:
            if mistakes_found >= num_mistakes:
                break

            predicted_sequences = predict_sequences(model, canvas_patches)
            
            # Compare each sequence in the batch
            for i in range(canvas_patches.size(0)):
                if mistakes_found >= num_mistakes:
                    break

                pred_seq = predicted_sequences[i].cpu().numpy()
                true_seq = target_outputs[i].cpu().numpy()
                
                # Filter out padding for comparison
                true_labels = [l for l in true_seq if l != PAD_TOKEN and l != EOS_TOKEN]
                pred_labels = [p for p in pred_seq if p != PAD_TOKEN and p != EOS_TOKEN and p != SOS_TOKEN]

                if true_labels != pred_labels:
                    print(f"  Mistake {mistakes_found + 1}:")
                    print(f"    - True sequence: {true_labels}")
                    print(f"    - Pred sequence: {pred_labels}")
                    
                    # Visualize the canvas that caused the mistake
                    visualize_canvas_sequence(canvas_patches.cpu(), (None, target_outputs.cpu()), sequence_idx=i)
                    
                    mistakes_found += 1
    
    if mistakes_found == 0:
        print("  No mistakes found in the analyzed batches!")
    print()


def show_learning_insights(history):
    """
    Provide key learning insights.
    """
    print("Key Learning Insights:")
    
    val_accs = history['val_accuracies']
    
    # Learning speed
    epochs_to_90 = None
    for i, acc in enumerate(val_accs):
        if acc >= 90.0:
            epochs_to_90 = i + 1
            break
    
    if epochs_to_90:
        print(f"  • Reached 90% accuracy in {epochs_to_90} epochs")
    
    # Convergence
    if len(history['train_losses']) > 3:
        final_losses = history['train_losses'][-3:]
        if np.var(final_losses) < 0.001:
            print("  • Training converged smoothly")
        else:
            print("  • Training still improving")
    
    # Transformer insights
    print("  • Vision Transformer successfully learned:")
    print("    - Patch-based image processing")
    print("    - Spatial relationships via self-attention")
    print("    - Feature extraction through multiple layers")
    
    best_acc = max(val_accs)
    if best_acc >= 95:
        print(f"  • Excellent performance achieved ({best_acc:.1f}%)")
    elif best_acc >= 90:
        print(f"  • Good performance achieved ({best_acc:.1f}%)")
    
    print() 