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

# Character-level FEN tokenizer
FEN_CHARS = "rnbqkbnrPPPPPPPP12345678/ wbkq-"
VOCAB = {char: idx + 1 for idx, char in enumerate(set(FEN_CHARS))}
PAD_IDX = 0
MAX_FEN_LENGTH = 100


def square_to_index(square_str):
    file_char = square_str[0]
    rank_char = square_str[1]
    file_idx = ord(file_char) - ord('a')
    rank_idx = int(rank_char) - 1
    return rank_idx * 8 + file_idx


def index_to_square(square_idx):
    return chess.square_name(square_idx)


def make_uci(start_idx, end_idx):
    return f"{index_to_square(start_idx)}{index_to_square(end_idx)}"


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
            "fen_str": fen_str,
            "uci_move": uci_move,
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


class BrahimClone(nn.Module):
    def __init__(self, vocab_size=len(VOCAB) + 1, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.start_head = nn.Linear(d_model, 64)
        self.end_head = nn.Linear(d_model, 64)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        memory = self.transformer_encoder(x)
        board_rep = memory.mean(dim=1)
        start_logits = self.start_head(board_rep)
        end_logits = self.end_head(board_rep)
        return start_logits, end_logits


def batch_metrics(start_logits, end_logits, start_targets, end_targets):
    start_pred = start_logits.argmax(dim=1)
    end_pred = end_logits.argmax(dim=1)

    start_correct = (start_pred == start_targets).sum().item()
    end_correct = (end_pred == end_targets).sum().item()
    exact_correct = ((start_pred == start_targets) & (end_pred == end_targets)).sum().item()

    return start_correct, end_correct, exact_correct


def run_epoch(model, dataloader, optimizer, device):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_start = 0
    total_end = 0
    total_exact = 0
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

    avg_loss = total_loss / max(total_samples, 1)
    start_acc = 100.0 * total_start / max(total_samples, 1)
    end_acc = 100.0 * total_end / max(total_samples, 1)
    exact_acc = 100.0 * total_exact / max(total_samples, 1)

    return avg_loss, start_acc, end_acc, exact_acc


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
    plt.savefig(os.path.join(plot_dir, "loss_curves.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_exact"], label="Train Exact")
    plt.plot(epochs, history["val_exact"], label="Val Exact")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Exact Move Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "exact_accuracy.png"), dpi=300)
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
            start_pred = start_logits.argmax(dim=1).cpu().tolist()
            end_pred = end_logits.argmax(dim=1).cpu().tolist()

            batch_fens = batch["fen_str"]
            batch_uci = batch["uci_move"]
            start_targets = start_targets.cpu().tolist()
            end_targets = end_targets.cpu().tolist()

            for i in range(len(batch_fens)):
                pred_uci = make_uci(start_pred[i], end_pred[i])
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
    epochs=10,
    batch_size=128,
    lr=3e-4,
    save_dir="models",
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

    model = BrahimClone().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_path = os.path.join(save_dir, "brahim_clone_10min_no_rules_best.pth")
    last_path = os.path.join(save_dir, "brahim_clone_10min_no_rules_last.pth")

    best_val_exact = -1.0

    for epoch in range(1, epochs + 1):
        train_loss, train_start, train_end, train_exact = run_epoch(
            model, train_loader, optimizer, device
        )
        val_loss, val_start, val_end, val_exact = run_epoch(
            model, val_loader, None, device
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"train_exact {train_exact:.2f}% | val_exact {val_exact:.2f}%"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_exact"].append(train_exact)
        history["val_exact"].append(val_exact)

        if val_exact > best_val_exact:
            best_val_exact = val_exact
            torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), last_path)

    plot_training_curves(history, plot_dir="plots")

    print("\nTesting best model...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_start, test_end, test_exact = run_epoch(
        model, test_loader, None, device
    )

    print(
        f"Test loss {test_loss:.4f} | "
        f"start_acc {test_start:.2f}% | end_acc {test_end:.2f}% | exact_acc {test_exact:.2f}%"
    )

    correct, wrong = collect_examples(model, test_loader, device)
    save_examples(correct, wrong, out_dir=os.path.join("plots", "qualitative"))


if __name__ == "__main__":
    train_csv = os.path.join("data", "splits", "brahim_10min_training_set.csv")
    val_csv = os.path.join("data", "splits", "brahim_10min_validation_set.csv")
    test_csv = os.path.join("data", "splits", "brahim_10min_test_set.csv")

    train_model(train_csv, val_csv, test_csv)
