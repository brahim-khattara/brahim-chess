import math
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import chess
import chess.svg

# Added the period '.' for empty squares
FEN_CHARS = "PNBRQKpnbrqk1234567890wb -/abcdefgh]."
UNIQUE_CHARS = sorted(list(set(FEN_CHARS)))

VOCAB = {char: idx + 1 for idx, char in enumerate(UNIQUE_CHARS)}
PAD_IDX = 0
MAX_FEN_LENGTH = 120


def square_to_index(square_str):
    file_char = square_str[0]
    rank_char = square_str[1]
    file_idx = ord(file_char) - ord("a")
    rank_idx = int(rank_char) - 1
    return rank_idx * 8 + file_idx


def index_to_square(square_idx):
    return chess.square_name(square_idx)


def make_uci(start_idx, end_idx):
    return f"{index_to_square(start_idx)}{index_to_square(end_idx)}"


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

        input_text = unpack_fen(fen_str)

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
            "uci_move": uci_move,
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


def batch_metrics(start_logits, end_logits, start_targets, end_targets):
    start_pred = start_logits.argmax(dim=1)
    end_pred = end_logits.argmax(dim=1)

    start_correct = (start_pred == start_targets).sum().item()
    end_correct = (end_pred == end_targets).sum().item()
    exact_correct = ((start_pred == start_targets) & (end_pred == end_targets)).sum().item()

    return start_correct, end_correct, exact_correct


def masked_exact_correct(start_logits, end_logits, fen_strs, start_targets, end_targets):
    start_probs = F.softmax(start_logits, dim=1)
    end_probs = F.softmax(end_logits, dim=1)

    exact_correct = 0
    for i in range(start_probs.size(0)):
        joint = torch.outer(start_probs[i], end_probs[i]).reshape(-1)
        mask = legal_move_mask(fen_strs[i]).to(joint.device)
        joint = joint * mask
        pred_flat = int(joint.argmax().item())
        pred_start = pred_flat // 64
        pred_end = pred_flat % 64

        if pred_start == start_targets[i] and pred_end == end_targets[i]:
            exact_correct += 1

    return exact_correct


def run_epoch(model, dataloader, optimizer, device, masked_eval=False):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_start = 0
    total_end = 0
    total_exact = 0
    total_exact_masked = 0
    total_samples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        start_targets = batch["start_target"].to(device)
        end_targets = batch["end_target"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        start_logits, end_logits = model(input_ids)
        loss_start = F.cross_entropy(start_logits, start_targets)
        loss_end = F.cross_entropy(end_logits, end_targets)
        loss = loss_start + loss_end

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size

        start_correct, end_correct, exact_correct = batch_metrics(
            start_logits, end_logits, start_targets, end_targets
        )
        total_start += start_correct
        total_end += end_correct
        total_exact += exact_correct
        total_samples += batch_size

        if masked_eval:
            exact_masked = masked_exact_correct(
                start_logits,
                end_logits,
                batch["fen_str"],
                start_targets.cpu().tolist(),
                end_targets.cpu().tolist(),
            )
            total_exact_masked += exact_masked

    avg_loss = total_loss / max(total_samples, 1)
    start_acc = 100.0 * total_start / max(total_samples, 1)
    end_acc = 100.0 * total_end / max(total_samples, 1)
    exact_acc = 100.0 * total_exact / max(total_samples, 1)
    exact_masked_acc = 0.0

    if masked_eval:
        exact_masked_acc = 100.0 * total_exact_masked / max(total_samples, 1)

    return avg_loss, start_acc, end_acc, exact_acc, exact_masked_acc


def plot_training_curves(history, plot_dir):
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "exp12_loss_curves.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_exact"], label="Train Exact (Unmasked)")
    plt.plot(epochs, history["val_exact"], label="Val Exact (Unmasked)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("EXP-12 Exact Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "exp12_exact_accuracy.png"), dpi=300)
    plt.close()


def render_board_svg(fen_str, predicted_uci, actual_uci, out_path):
    board = chess.Board(fen_str)
    arrows = []

    try:
        pred_move = chess.Move.from_uci(predicted_uci)
        arrows.append(chess.svg.Arrow(pred_move.from_square, pred_move.to_square, color="#d62728"))
    except ValueError:
        pred_move = None

    if actual_uci is not None:
        try:
            actual_move = chess.Move.from_uci(actual_uci)
            arrows.append(chess.svg.Arrow(actual_move.from_square, actual_move.to_square, color="#2ca02c"))
        except ValueError:
            actual_move = None

    svg = chess.svg.board(board=board, arrows=arrows, size=400, coordinates=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)


def collect_examples(model, dataloader, device, max_correct=6, max_wrong=6):
    model.eval()
    correct = []
    wrong = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            start_targets = batch["start_target"].to(device)
            end_targets = batch["end_target"].to(device)

            start_logits, end_logits = model(input_ids)
            start_probs = F.softmax(start_logits, dim=1)
            end_probs = F.softmax(end_logits, dim=1)

            batch_fens = batch["fen_str"]
            batch_uci = batch["uci_move"]
            start_targets = start_targets.cpu().tolist()
            end_targets = end_targets.cpu().tolist()

            for i in range(len(batch_fens)):
                joint = torch.outer(start_probs[i], end_probs[i]).reshape(-1)
                mask = legal_move_mask(batch_fens[i]).to(joint.device)
                joint = joint * mask
                pred_flat = int(joint.argmax().item())
                pred_start = pred_flat // 64
                pred_end = pred_flat % 64
                pred_uci = make_uci(pred_start, pred_end)

                actual_uci = batch_uci[i]
                actual_short = actual_uci[:4]

                if pred_uci == actual_short and len(correct) < max_correct:
                    correct.append((batch_fens[i], pred_uci, actual_short))
                elif pred_uci != actual_short and len(wrong) < max_wrong:
                    wrong.append((batch_fens[i], pred_uci, actual_short))

                if len(correct) >= max_correct and len(wrong) >= max_wrong:
                    return correct, wrong

    return correct, wrong


def save_examples(correct, wrong, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for i, (fen, pred, actual) in enumerate(correct, start=1):
        out_path = os.path.join(out_dir, f"correct_{i:02d}.svg")
        render_board_svg(fen, pred, actual, out_path)

    for i, (fen, pred, actual) in enumerate(wrong, start=1):
        out_path = os.path.join(out_dir, f"wrong_{i:02d}.svg")
        render_board_svg(fen, pred, actual, out_path)


def train_model(
    train_csv,
    val_csv,
    test_csv,
    epochs=300,
    batch_size=128,
    lr=1e-4,
    save_dir="models",
    patience=8  # Early stopping patience
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Training on device: {device}")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_exact": [],
        "val_exact": [],
    }

    train_dataset = ChessBehaviorDataset(train_csv)
    val_dataset = ChessBehaviorDataset(val_csv)
    test_dataset = ChessBehaviorDataset(test_csv)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BrahimClone2D().to(device)
    
    # 1e-4 LR with slightly higher weight decay for stability
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # The Scheduler: Cuts LR in half if validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_path = os.path.join(save_dir, "brahim_clone_exp12_best.pth")
    last_path = os.path.join(save_dir, "brahim_clone_exp12_last.pth")

    best_val_exact = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        # FAST MODE: masked_eval is OFF during the epoch loop
        train_loss, _, _, train_exact, _ = run_epoch(model, train_loader, optimizer, device, masked_eval=False)
        val_loss, _, _, val_exact, _ = run_epoch(model, val_loader, None, device, masked_eval=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_exact"].append(train_exact)
        history["val_exact"].append(val_exact)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"val_exact {val_exact:.2f}%"
        )

        # Step the scheduler
        scheduler.step(val_loss)

        # Early Stopping based on raw accuracy
        if val_exact > best_val_exact:
            best_val_exact = val_exact
            torch.save(model.state_dict(), best_path)
            epochs_without_improvement = 0
            print("  -> New best model saved!")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    torch.save(model.state_dict(), last_path)

    # REMINDER: You MUST delete the "val_exact_masked" plot line from plot_training_curves!
    # plot_training_curves(history, plot_dir="plots")

    print("\nTesting best model (Calculating mask, please wait)...")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    
    # FINAL EXAM: masked_eval is ON
    test_loss, _, _, test_exact, test_exact_masked = run_epoch(
        model, test_loader, None, device, masked_eval=True
    )

    print(
        f"FINAL RESULTS: Test loss {test_loss:.4f} | "
        f"exact {test_exact:.2f}% | exact_masked {test_exact_masked:.2f}%"
    )

    correct, wrong = collect_examples(model, test_loader, device)
    save_examples(correct, wrong, out_dir=os.path.join("plots", "exp12_qualitative"))

if __name__ == "__main__":
    train_csv = os.path.join("data", "splits", "brahim_10min_elo1100_training_set.csv")
    val_csv = os.path.join("data", "splits", "brahim_10min_elo1100_validation_set.csv")
    test_csv = os.path.join("data", "splits", "brahim_10min_elo1100_test_set.csv")

    train_model(train_csv, val_csv, test_csv)
