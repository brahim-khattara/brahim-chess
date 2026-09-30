import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import chess

# Fixed, deterministic character-level FEN tokenizer
FEN_CHARS = "PNBRQKpnbrqk12345678wb -/"
UNIQUE_CHARS = sorted(list(set(FEN_CHARS))) 

VOCAB = {char: idx + 1 for idx, char in enumerate(UNIQUE_CHARS)}
PAD_IDX = 0
MAX_FEN_LENGTH = 100

def square_to_index(square_str):
    file_char = square_str[0]
    rank_char = square_str[1]
    file_idx = ord(file_char) - ord('a')
    rank_idx = int(rank_char) - 1
    return rank_idx * 8 + file_idx

def legal_move_mask(fen_str):
    board = chess.Board(fen_str)
    mask = torch.zeros(4096, dtype=torch.float32)
    for move in board.legal_moves:
        flat_idx = (move.from_square * 64) + move.to_square
        mask[flat_idx] = 1.0
    return mask

class ChessBehaviorDataset(Dataset):
    def __init__(self, csv_filepath):
        print(f"Loading data from {csv_filepath}...")
        self.data = pd.read_csv(csv_filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen_str = row["Board_State_FEN"]
        uci_move = row["Brahim_Move_UCI"]

        tokens = [VOCAB.get(c, PAD_IDX) for c in fen_str]
        if len(tokens) < MAX_FEN_LENGTH:
            tokens += [PAD_IDX] * (MAX_FEN_LENGTH - len(tokens))
        else:
            tokens = tokens[:MAX_FEN_LENGTH]

        start_sq = square_to_index(uci_move[:2])
        end_sq = square_to_index(uci_move[2:4])

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "start_target": torch.tensor(start_sq, dtype=torch.long),
            "end_target": torch.tensor(end_sq, dtype=torch.long),
            "fen_str": fen_str,  # Added this so we can calculate the mask
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_FEN_LENGTH):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0)]

class BrahimCloneCoupled(nn.Module):
    def __init__(self, vocab_size=len(VOCAB) + 1, d_model=512, nhead=8, num_layers=6, start_embed_dim=32):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.start_head = nn.Linear(d_model, 64)
        self.start_embed = nn.Embedding(64, start_embed_dim)
        self.end_head = nn.Linear(d_model + start_embed_dim, 64)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        memory = self.transformer_encoder(x)
        board_rep = memory.mean(dim=1)

        start_logits = self.start_head(board_rep)
        start_probs = F.softmax(start_logits, dim=1)
        start_emb = start_probs @ self.start_embed.weight

        end_input = torch.cat([board_rep, start_emb], dim=1)
        end_logits = self.end_head(end_input)
        return start_logits, end_logits

def load_model(weights_path, device):
    model = BrahimCloneCoupled().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def eval_topk_masked(model, dataloader, device, k=5):
    total_samples = 0
    total_top1 = 0
    total_topk = 0

    print("Evaluating masked Top-K (this will take a moment to calculate masks)...")
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            start_targets = batch["start_target"].cpu().tolist()
            end_targets = batch["end_target"].cpu().tolist()
            fen_strs = batch["fen_str"]

            start_logits, end_logits = model(input_ids)
            start_probs = F.softmax(start_logits, dim=1)
            end_probs = F.softmax(end_logits, dim=1)

            batch_size = input_ids.size(0)

            for i in range(batch_size):
                # 1. Create the 4096-flat joint probability array
                joint = torch.outer(start_probs[i], end_probs[i]).reshape(-1)
                
                # 2. Apply the mask
                mask = legal_move_mask(fen_strs[i]).to(joint.device)
                masked_joint = joint * mask
                
                # 3. Calculate target flat index
                target_flat = (start_targets[i] * 64) + end_targets[i]

                # Top-1 Check
                pred_top1 = int(masked_joint.argmax().item())
                if pred_top1 == target_flat:
                    total_top1 += 1

                # Top-K Check
                # Grab the indices of the K highest probabilities
                topk_indices = torch.topk(masked_joint, k=k).indices.tolist()

                if target_flat in topk_indices:
                    total_topk += 1

            total_samples += batch_size

    top1_acc = 100.0 * total_top1 / max(total_samples, 1)
    topk_acc = 100.0 * total_topk / max(total_samples, 1)
    return top1_acc, topk_acc

if __name__ == "__main__":
    weights_path = os.path.join("models", "brahim_clone_exp12_best.pth")
    test_csv = os.path.join("data", "splits", "brahim_10min_elo1100_test_set.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = ChessBehaviorDataset(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = load_model(weights_path, device)

    top1_acc, top5_acc = eval_topk_masked(model, test_loader, device, k=5)
    delta = top5_acc - top1_acc

    print(f"\n--- EXP-10 (Masked Evaluation) ---")
    print(f"Top-1 Masked Exact Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Masked Exact Accuracy: {top5_acc:.2f}%")
    print(f"Delta (Top-5 - Top-1): {delta:.2f}%")