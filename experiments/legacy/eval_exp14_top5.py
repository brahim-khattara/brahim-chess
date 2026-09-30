import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import chess

# Fixed, deterministic character-level FEN tokenizer
FEN_CHARS = "PNBRQKpnbrqk1234567890wb -/abcdefgh]."
UNIQUE_CHARS = sorted(list(set(FEN_CHARS))) 

VOCAB = {char: idx + 1 for idx, char in enumerate(UNIQUE_CHARS)}
PAD_IDX = 0
MAX_FEN_LENGTH = 120

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

def unpack_fen(fen_str, last_move_uci="none"):
    parts = fen_str.split(" ")
    board = parts[0]
    metadata = " ".join(parts[1:])

    ranks = board.split("/")
    ranks.reverse()

    unpacked = ""
    for rank in ranks:
        for char in rank:
            if char.isdigit():
                unpacked += "." * int(char)
            else:
                unpacked += char

    last_move = last_move_uci if isinstance(last_move_uci, str) else "none"
    padded_history = last_move.strip().ljust(4, " ") + "]"

    return f"{unpacked} {metadata} {padded_history}"

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
        last_move = row.get("Opponent_Last_Move_UCI", "")
        if pd.isna(last_move):
            last_move = ""

        # Unpack the FEN with historical context!
        input_text = unpack_fen(fen_str, last_move)

        tokens = [VOCAB.get(c, PAD_IDX) for c in input_text]
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
            "fen_str": fen_str,  
        }

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model=512, max_len=MAX_FEN_LENGTH):
        super().__init__()
        self.file_embed = nn.Embedding(8, d_model // 2)
        self.rank_embed = nn.Embedding(8, d_model // 2)
        self.meta_embed = nn.Embedding(max_len, d_model)

        files = torch.zeros(64, dtype=torch.long)
        ranks = torch.zeros(64, dtype=torch.long)
        for i in range(64):
            files[i] = i % 8
            ranks[i] = i // 8

        self.register_buffer("files", files)
        self.register_buffer("ranks", ranks)

    def forward(self, x):
        seq_len = x.size(1)

        emb_x = self.file_embed(self.files)
        emb_y = self.rank_embed(self.ranks)
        board_pe = torch.cat([emb_x, emb_y], dim=1)

        if seq_len <= 64:
            return x + board_pe[:seq_len].unsqueeze(0)

        meta_indices = torch.arange(seq_len - 64, device=x.device) + 64
        meta_pe = self.meta_embed(meta_indices)

        full_pe = torch.cat([board_pe, meta_pe], dim=0)
        return x + full_pe.unsqueeze(0)

class BrahimClone2D(nn.Module):
    def __init__(self, vocab_size=len(VOCAB) + 1, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding2D(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.context_norm = nn.LayerNorm(d_model)

        self.start_head = nn.Linear(d_model, 1)
        self.end_head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x)

        board_tokens = memory[:, :64, :]

        start_logits = self.start_head(board_tokens).squeeze(-1)
        start_probs = F.softmax(start_logits, dim=1)

        start_emb = start_probs.unsqueeze(-1) * board_tokens
        start_context = start_emb.sum(dim=1).unsqueeze(1)

        end_input = board_tokens + start_context
        end_input = self.context_norm(end_input)
        end_logits = self.end_head(end_input).squeeze(-1)

        return start_logits, end_logits

def load_model(weights_path, device):
    model = BrahimClone2D().to(device)
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
                joint = torch.outer(start_probs[i], end_probs[i]).reshape(-1)
                mask = legal_move_mask(fen_strs[i]).to(joint.device)
                masked_joint = joint * mask
                target_flat = (start_targets[i] * 64) + end_targets[i]

                pred_top1 = int(masked_joint.argmax().item())
                if pred_top1 == target_flat:
                    total_top1 += 1

                topk_indices = torch.topk(masked_joint, k=k).indices.tolist()

                if target_flat in topk_indices:
                    total_topk += 1

            total_samples += batch_size

    top1_acc = 100.0 * total_top1 / max(total_samples, 1)
    topk_acc = 100.0 * total_topk / max(total_samples, 1)
    return top1_acc, topk_acc

if __name__ == "__main__":
    weights_path = os.path.join("models", "brahim_clone_exp15_finetuned_best.pth")
    test_csv = os.path.join("data", "splits", "brahim_with_history_10min_elo1400_test_set.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = ChessBehaviorDataset(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = load_model(weights_path, device)

    top1_acc, top5_acc = eval_topk_masked(model, test_loader, device, k=5)
    delta = top5_acc - top1_acc

    print(f"\n--- EXP-14 Peak Self (Masked Evaluation) ---")
    print(f"Top-1 Masked Exact Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Masked Exact Accuracy: {top5_acc:.2f}%")
    print(f"Delta (Top-5 - Top-1): {delta:.2f}%")