import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
import chess
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 1. The Vocabulary
# We map every possible character in a FEN string to a unique integer.
FEN_CHARS = "rnbqkbnrPPPPPPPP12345678/ wbkq-"
VOCAB = {char: idx + 1 for idx, char in enumerate(set(FEN_CHARS))}
PAD_IDX = 0 # Used to ensure all FEN strings in a batch are the exact same length
MAX_FEN_LENGTH = 100 

def square_to_index(square_str):
    """Converts a square like 'e2' to an integer from 0 to 63."""
    file_char = square_str[0]
    rank_char = square_str[1]
    file_idx = ord(file_char) - ord('a')
    rank_idx = int(rank_char) - 1
    return rank_idx * 8 + file_idx

def get_legal_move_mask(fen_str, device):
    """
    Generate a legal move mask for a given FEN position.
    Returns a [4096] tensor where 1 = legal move, 0 = illegal move.
    """
    try:
        board = chess.Board(fen_str)
        mask = torch.zeros(4096, device=device)
        
        for move in board.legal_moves:
            start_sq = move.from_square  # Already 0-63
            end_sq = move.to_square      # Already 0-63
            flat_idx = (start_sq * 64) + end_sq
            mask[flat_idx] = 1.0
        
        return mask
    except:
        # If FEN is invalid, return all ones (allow all moves)
        return torch.ones(4096, device=device)

class ChessBehaviorDataset(Dataset):
    def __init__(self, csv_filepath):
        print(f"Loading 110,000+ moves from {csv_filepath}...")
        self.data = pd.read_csv(csv_filepath)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen_str = row['Board_State_FEN']
        uci_move = row['Brahim_Move_UCI']
        
        # Tokenize the FEN string
        tokens = [VOCAB.get(c, PAD_IDX) for c in fen_str]
        # Pad sequence to a fixed length of 100 for batching
        if len(tokens) < MAX_FEN_LENGTH:
            tokens += [PAD_IDX] * (MAX_FEN_LENGTH - len(tokens))
        else:
            tokens = tokens[:MAX_FEN_LENGTH]
            
        # Parse the UCI move (e.g., 'e2e4') into Start and End heads
        # We slice [:2] and [2:4] to safely ignore promotion characters like 'q'
        start_sq = square_to_index(uci_move[:2])
        end_sq = square_to_index(uci_move[2:4])
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'start_target': torch.tensor(start_sq, dtype=torch.long),
            'end_target': torch.tensor(end_sq, dtype=torch.long)
        }
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_FEN_LENGTH):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, embedding_dim]
        return x + self.pe[:x.size(0)]

def calculate_topk_joint_accuracy(start_logits, end_logits, start_targets, end_targets):
    """
    Calculate Top-1, Top-3, and Top-5 Joint (Exact Match) Accuracy.
    
    Args:
        start_logits: [batch_size, 64]
        end_logits: [batch_size, 64]
        start_targets: [batch_size]
        end_targets: [batch_size]
    
    Returns:
        tuple: (top1_correct, top3_correct, top5_correct)
    """
    batch_size = start_logits.size(0)
    
    # 1. Apply softmax to both logits
    start_probs = F.softmax(start_logits, dim=1)  # [batch_size, 64]
    end_probs = F.softmax(end_logits, dim=1)      # [batch_size, 64]
    
    # 2. Calculate joint probability matrix using torch.bmm()
    # Reshape for bmm: [batch_size, 64, 1] and [batch_size, 1, 64]
    start_probs_reshaped = start_probs.unsqueeze(2)  # [batch_size, 64, 1]
    end_probs_reshaped = end_probs.unsqueeze(1)      # [batch_size, 1, 64]
    
    # Joint matrix: [batch_size, 64, 64]
    joint_probs = torch.bmm(start_probs_reshaped, end_probs_reshaped)
    
    # 3. Flatten the joint probability matrix to [batch_size, 4096]
    joint_probs_flat = joint_probs.reshape(batch_size, -1)
    
    # 4. Flatten the targets: (start_targets * 64) + end_targets
    flattened_targets = (start_targets * 64) + end_targets
    
    # 5. Use torch.topk to get the top 5 predicted indices
    topk_values, topk_indices = torch.topk(joint_probs_flat, k=5, dim=1)
    
    # 6. Create boolean masks for top-1, top-3, top-5
    top1_mask = (topk_indices[:, 0] == flattened_targets)
    top3_mask = (topk_indices[:, :3] == flattened_targets.unsqueeze(1)).any(dim=1)
    top5_mask = (topk_indices[:, :5] == flattened_targets.unsqueeze(1)).any(dim=1)
    
    # 7. Sum the True values
    top1_correct = top1_mask.sum().item()
    top3_correct = top3_mask.sum().item()
    top5_correct = top5_mask.sum().item()
    
    return top1_correct, top3_correct, top5_correct

def plot_training_metrics(history, plot_dir="training_plots"):
    """
    Generate comprehensive training visualizations.
    
    Args:
        history: dict containing lists of metrics per epoch
        plot_dir: directory to save plots
    """
    # Create plot directory
    Path(plot_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Loss Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 's-', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss Detail', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '01_loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Exact Move Accuracy
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(epochs, history['train_exact_acc'], 'o-', label='Train Exact Move Accuracy', linewidth=2.5, markersize=5)
    ax.plot(epochs, history['val_exact_acc'], 's-', label='Validation Exact Move Accuracy', linewidth=2.5, markersize=5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Exact Move Accuracy (Both Start & End Correct)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '02_exact_move_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Component Accuracies (Start & End)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, history['train_start_acc'], 'o-', label='Train Start Square', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_start_acc'], 's-', label='Validation Start Square', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Start Square Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    ax2.plot(epochs, history['train_end_acc'], 'o-', label='Train End Square', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_end_acc'], 's-', label='Validation End Square', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('End Square Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '03_component_accuracies.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Top-K Accuracies
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(epochs, history['train_top1_acc'], 'o-', label='Top-1 Accuracy', linewidth=2.5, markersize=5)
    ax.plot(epochs, history['train_top3_acc'], 's-', label='Top-3 Accuracy', linewidth=2.5, markersize=5)
    ax.plot(epochs, history['train_top5_acc'], '^-', label='Top-5 Accuracy', linewidth=2.5, markersize=5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Top-K Joint Accuracy (Training)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '04_topk_train_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Top-K Accuracies Validation
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(epochs, history['val_top1_acc'], 'o-', label='Top-1 Accuracy', linewidth=2.5, markersize=5)
    ax.plot(epochs, history['val_top3_acc'], 's-', label='Top-3 Accuracy', linewidth=2.5, markersize=5)
    ax.plot(epochs, history['val_top5_acc'], '^-', label='Top-5 Accuracy', linewidth=2.5, markersize=5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Top-K Joint Accuracy (Validation)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '05_topk_val_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. All Metrics Comparison (Train)
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(epochs, history['train_exact_acc'], 'o-', label='Exact Move', linewidth=2.5, markersize=5)
    ax.plot(epochs, history['train_start_acc'], 's-', label='Start Square', linewidth=2, markersize=4)
    ax.plot(epochs, history['train_end_acc'], '^-', label='End Square', linewidth=2, markersize=4)
    ax.plot(epochs, history['train_top1_acc'], 'd-', label='Top-1 Joint', linewidth=2, markersize=4)
    ax.plot(epochs, history['train_top3_acc'], 'x-', label='Top-3 Joint', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('All Training Metrics Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '06_all_train_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. All Metrics Comparison (Validation)
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(epochs, history['val_exact_acc'], 'o-', label='Exact Move', linewidth=2.5, markersize=5)
    ax.plot(epochs, history['val_start_acc'], 's-', label='Start Square', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_end_acc'], '^-', label='End Square', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_top1_acc'], 'd-', label='Top-1 Joint', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_top3_acc'], 'x-', label='Top-3 Joint', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('All Validation Metrics Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '07_all_val_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Train vs Validation Exact Move Accuracy Gap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    train_val_gap = [t - v for t, v in zip(history['train_exact_acc'], history['val_exact_acc'])]
    colors = ['red' if gap > 0 else 'green' for gap in train_val_gap]
    
    ax.bar(epochs, train_val_gap, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gap (%)', fontsize=12)
    ax.set_title('Overfitting Gap (Train - Validation Exact Move Accuracy)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '08_overfitting_gap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ All plots saved to '{plot_dir}/' directory")
    print(f"  - 01_loss_curves.png")
    print(f"  - 02_exact_move_accuracy.png")
    print(f"  - 03_component_accuracies.png")
    print(f"  - 04_topk_train_accuracy.png")
    print(f"  - 05_topk_val_accuracy.png")
    print(f"  - 06_all_train_metrics.png")
    print(f"  - 07_all_val_metrics.png")
    print(f"  - 08_overfitting_gap.png")

class BrahimClone(nn.Module):
    def __init__(self, vocab_size=len(VOCAB)+1, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # We pool the transformer output by taking the mean across the sequence
        # to get a single vector representing the entire board state
        
        # Head A: Which piece to move?
        self.start_head = nn.Linear(d_model, 64)
        # Head B: Where does it go?
        self.end_head = nn.Linear(d_model, 64)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        
        # 1. Embed and add spatial awareness
        x = self.embedding(x) * math.sqrt(self.d_model) 
        
        # Transformer expects [seq_len, batch_size, d_model] if batch_first=False, 
        # but we set batch_first=True, so it expects [batch_size, seq_len, d_model]
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # 2. Pass through the 6-layer brain
        memory = self.transformer_encoder(x)
        
        # 3. Pool the context (average the tokens)
        board_representation = memory.mean(dim=1) 
        
        # 4. Output the two 64-class probability distributions
        start_logits = self.start_head(board_representation)
        end_logits = self.end_head(board_representation)
        
        return start_logits, end_logits
    

def train_model(train_csv, val_csv, test_csv, epochs=30, batch_size=128, lr=0.0003, use_legal_masking=True):
    # Setup Device (Use GPU if available, otherwise Apple Silicon MPS or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Using Legal Move Masking: {use_legal_masking}")

    # Initialize history tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_start_acc': [],
        'train_end_acc': [],
        'train_exact_acc': [],
        'train_top1_acc': [],
        'train_top3_acc': [],
        'train_top5_acc': [],
        'val_start_acc': [],
        'val_end_acc': [],
        'val_exact_acc': [],
        'val_top1_acc': [],
        'val_top3_acc': [],
        'val_top5_acc': [],
    }

    # Load Data
    print(f"Loading training data from {train_csv}...")
    train_dataset = ChessBehaviorDataset(train_csv)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Loading validation data from {val_csv}...")
    val_dataset = ChessBehaviorDataset(val_csv)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Loading test data from {test_csv}...")
    test_dataset = ChessBehaviorDataset(test_csv)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model
    model = BrahimClone().to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize Optimizer and Loss Functions
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_exact_acc = 0.0
    best_epoch = 0

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_starts = 0
        correct_ends = 0
        correct_exact_moves = 0
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            start_targets = batch['start_target'].to(device)
            end_targets = batch['end_target'].to(device)

            # Forward Pass
            optimizer.zero_grad()
            start_logits, end_logits = model(input_ids)

            # Path B: Legal Move Masking
            if use_legal_masking:
                batch_size_cur = input_ids.size(0)
                for i in range(batch_size_cur):
                    try:
                        fen_str = train_dataset.data.iloc[batch_idx * batch_size + i]['Board_State_FEN']
                        board = chess.Board(fen_str)
                        
                        # Get legal moves
                        legal_starts = set()
                        legal_ends = set()
                        for move in board.legal_moves:
                            legal_starts.add(move.from_square)
                            legal_ends.add(move.to_square)
                        
                        # Mask illegal moves in logits - set them to very negative values
                        for sq in range(64):
                            if sq not in legal_starts:
                                start_logits[i, sq] = float('-inf')
                            if sq not in legal_ends:
                                end_logits[i, sq] = float('-inf')
                    except:
                        pass  # If FEN is invalid, skip masking

            # Calculate Dual Loss
            loss_start = criterion(start_logits, start_targets)
            loss_end = criterion(end_logits, end_targets)
            loss = loss_start + loss_end

            # Backward Pass
            loss.backward()
            optimizer.step()

            # Tracking Metrics
            total_loss += loss.item()
            
            # Calculate accuracy for this batch
            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)
            
            correct_starts += (start_preds == start_targets).sum().item()
            correct_ends += (end_preds == end_targets).sum().item()
            
            # Correct way: calculate exact move matches using logical AND per sample
            exact_matches = ((start_preds == start_targets) & (end_preds == end_targets)).sum().item()
            correct_exact_moves += exact_matches
            
            # Calculate Top-1, Top-3, Top-5 Joint Accuracy
            top1, top3, top5 = calculate_topk_joint_accuracy(start_logits, end_logits, start_targets, end_targets)
            correct_top1 += top1
            correct_top3 += top3
            correct_top5 += top5
            
            total_samples += input_ids.size(0)

            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_dataloader)} | Loss: {loss.item():.4f}")

        # Epoch Summaries
        avg_loss = total_loss / len(train_dataloader)
        start_acc = (correct_starts / total_samples) * 100
        end_acc = (correct_ends / total_samples) * 100
        exact_move_acc = (correct_exact_moves / total_samples) * 100
        top1_acc = (correct_top1 / total_samples) * 100
        top3_acc = (correct_top3 / total_samples) * 100
        top5_acc = (correct_top5 / total_samples) * 100
        
        # Store history
        history['train_loss'].append(avg_loss)
        history['train_start_acc'].append(start_acc)
        history['train_end_acc'].append(end_acc)
        history['train_exact_acc'].append(exact_move_acc)
        history['train_top1_acc'].append(top1_acc)
        history['train_top3_acc'].append(top3_acc)
        history['train_top5_acc'].append(top5_acc)
        
        print(f"\n--- Epoch {epoch+1} Training Summary ---")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Start Square Accuracy: {start_acc:.2f}%")
        print(f"End Square Accuracy: {end_acc:.2f}%")
        print(f"Exact Move Accuracy (Both Correct): {exact_move_acc:.2f}%")
        print(f"Top-1 Joint Accuracy: {top1_acc:.2f}%")
        print(f"Top-3 Joint Accuracy: {top3_acc:.2f}%")
        print(f"Top-5 Joint Accuracy: {top5_acc:.2f}%")
        print("---------------------------------------\n")
        
        # Validation
        print("Running validation...")
        model.eval()
        val_loss = 0
        val_exact_moves = 0
        val_top1 = 0
        val_top3 = 0
        val_top5 = 0
        val_start_correct = 0
        val_end_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                start_targets = batch['start_target'].to(device)
                end_targets = batch['end_target'].to(device)
                
                start_logits, end_logits = model(input_ids)
                
                loss_start = criterion(start_logits, start_targets)
                loss_end = criterion(end_logits, end_targets)
                val_loss += (loss_start + loss_end).item()
                
                start_preds = torch.argmax(start_logits, dim=1)
                end_preds = torch.argmax(end_logits, dim=1)
                
                val_start_correct += (start_preds == start_targets).sum().item()
                val_end_correct += (end_preds == end_targets).sum().item()
                
                exact_matches = ((start_preds == start_targets) & (end_preds == end_targets)).sum().item()
                val_exact_moves += exact_matches
                
                top1, top3, top5 = calculate_topk_joint_accuracy(start_logits, end_logits, start_targets, end_targets)
                val_top1 += top1
                val_top3 += top3
                val_top5 += top5
                
                val_total += input_ids.size(0)
        
        val_avg_loss = val_loss / len(val_dataloader)
        val_start_acc = (val_start_correct / val_total) * 100
        val_end_acc = (val_end_correct / val_total) * 100
        val_exact_acc = (val_exact_moves / val_total) * 100
        val_top1_acc = (val_top1 / val_total) * 100
        val_top3_acc = (val_top3 / val_total) * 100
        val_top5_acc = (val_top5 / val_total) * 100
        
        # Store history
        history['val_loss'].append(val_avg_loss)
        history['val_start_acc'].append(val_start_acc)
        history['val_end_acc'].append(val_end_acc)
        history['val_exact_acc'].append(val_exact_acc)
        history['val_top1_acc'].append(val_top1_acc)
        history['val_top3_acc'].append(val_top3_acc)
        history['val_top5_acc'].append(val_top5_acc)
        
        print(f"--- Epoch {epoch+1} Validation Summary ---")
        print(f"Validation Loss: {val_avg_loss:.4f}")
        print(f"Start Square Accuracy: {val_start_acc:.2f}%")
        print(f"End Square Accuracy: {val_end_acc:.2f}%")
        print(f"Exact Move Accuracy: {val_exact_acc:.2f}%")
        print(f"Top-1 Joint Accuracy: {val_top1_acc:.2f}%")
        print(f"Top-3 Joint Accuracy: {val_top3_acc:.2f}%")
        print(f"Top-5 Joint Accuracy: {val_top5_acc:.2f}%")
        
        # Save best model
        if val_exact_acc > best_val_exact_acc:
            best_val_exact_acc = val_exact_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "brahim_clone_10min_best.pth")
            print(f"✓ Best model saved! (Epoch {best_epoch})")
        
        print("---------------------------------------\n")

    # Save the finalized behavioral clone
    torch.save(model.state_dict(), "brahim_clone_10min.pth")
    print(f"Training Complete! Best model was at Epoch {best_epoch} with {best_val_exact_acc:.2f}% exact accuracy.\n")
    
    # Generate visualizations
    plot_training_metrics(history)
    
    # Test Evaluation
    print("="*50)
    print("Running final test evaluation...")
    model.eval()
    test_loss = 0
    test_exact_moves = 0
    test_top1 = 0
    test_top3 = 0
    test_top5 = 0
    test_start_correct = 0
    test_end_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            start_targets = batch['start_target'].to(device)
            end_targets = batch['end_target'].to(device)
            
            start_logits, end_logits = model(input_ids)
            
            loss_start = criterion(start_logits, start_targets)
            loss_end = criterion(end_logits, end_targets)
            test_loss += (loss_start + loss_end).item()
            
            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)
            
            test_start_correct += (start_preds == start_targets).sum().item()
            test_end_correct += (end_preds == end_targets).sum().item()
            
            exact_matches = ((start_preds == start_targets) & (end_preds == end_targets)).sum().item()
            test_exact_moves += exact_matches
            
            top1, top3, top5 = calculate_topk_joint_accuracy(start_logits, end_logits, start_targets, end_targets)
            test_top1 += top1
            test_top3 += top3
            test_top5 += top5
            
            test_total += input_ids.size(0)
    
    test_avg_loss = test_loss / len(test_dataloader)
    test_start_acc = (test_start_correct / test_total) * 100
    test_end_acc = (test_end_correct / test_total) * 100
    test_exact_acc = (test_exact_moves / test_total) * 100
    test_top1_acc = (test_top1 / test_total) * 100
    test_top3_acc = (test_top3 / test_total) * 100
    test_top5_acc = (test_top5 / test_total) * 100
    
    print(f"--- Final Test Results ---")
    print(f"Test Loss: {test_avg_loss:.4f}")
    print(f"Start Square Accuracy: {test_start_acc:.2f}%")
    print(f"End Square Accuracy: {test_end_acc:.2f}%")
    print(f"Exact Move Accuracy: {test_exact_acc:.2f}%")
    print(f"Top-1 Joint Accuracy: {test_top1_acc:.2f}%")
    print(f"Top-3 Joint Accuracy: {test_top3_acc:.2f}%")
    print(f"Top-5 Joint Accuracy: {test_top5_acc:.2f}%")
    print("="*50)

# Execution
# You can set use_legal_masking=False to train without legal move constraints (Path A only)
train_model(
    "brahim_10min_training_set.csv",
    "brahim_10min_validation_set.csv",
    "brahim_10min_test_set.csv",
    use_legal_masking=True  # Path B: Enable legal move masking
)
