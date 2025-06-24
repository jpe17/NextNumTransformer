"""
Simple Training Loop
==================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, pad_token_id=12):
    print(f"Training for {epochs} epochs with lr={lr}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # History tracking
    history = {
        'train_losses': [], 'val_losses': [],
        'train_accuracies': [], 'val_accuracies': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        
        running_loss, running_correct, running_total = 0, 0, 0
        
        # tqdm progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (patches, decoder_inputs, target_outputs) in enumerate(pbar):
            optimizer.zero_grad()
            logits = model(patches, decoder_inputs)
            
            loss = criterion(logits.view(-1, logits.size(-1)), target_outputs.view(-1))
            loss.backward()
            optimizer.step()
            
            # Update running stats
            running_loss += loss.item()
            
            mask = target_outputs != pad_token_id
            predicted = logits.argmax(dim=-1)
            running_correct += (predicted[mask] == target_outputs[mask]).sum().item()
            running_total += mask.sum().item()
            
            # Show first batch tensor flow
            if epoch == 0 and batch_idx == 0:
                print("--- First batch tensor flow ---")
            
            # Update progress bar with current stats
            curr_loss = running_loss / (batch_idx + 1)
            curr_acc = 100 * running_correct / running_total if running_total > 0 else 0
            pbar.set_postfix({'Loss': f'{curr_loss:.4f}', 'Acc': f'{curr_acc:.1f}%'})
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for patches, decoder_inputs, target_outputs in val_loader:
                logits = model(patches, decoder_inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), target_outputs.view(-1))
                
                val_loss += loss.item()
                
                mask = target_outputs != pad_token_id
                predicted = logits.argmax(dim=-1)
                val_correct += (predicted[mask] == target_outputs[mask]).sum().item()
                val_total += mask.sum().item()
        
        # Final epoch stats
        final_train_acc = 100 * running_correct / running_total if running_total > 0 else 0
        final_val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        final_train_loss = running_loss / len(train_loader)
        final_val_loss = val_loss / len(val_loader)
        
        # Store history
        history['train_losses'].append(final_train_loss)
        history['val_losses'].append(final_val_loss)
        history['train_accuracies'].append(final_train_acc)
        history['val_accuracies'].append(final_val_acc)
        
        print(f"→ Train: {final_train_loss:.4f} ({final_train_acc:.1f}%) | "
              f"Val: {final_val_loss:.4f} ({final_val_acc:.1f}%)")
    
    print("\n🎉 Training complete!")
    return history 