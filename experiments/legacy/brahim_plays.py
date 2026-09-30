import torch
import torch.nn.functional as F
import chess
import numpy as np
import time

# --- 1. Copy the Architecture and Vocab here ---
# (You will need to paste the FEN_CHARS, VOCAB, square_to_index, 
# PositionalEncoding, and BrahimClone class from our training script here so it can load the weights).

def index_to_square(idx):
    """Converts a 0-63 integer back to a square like 'e2'"""
    file_char = chr(ord('a') + (idx % 8))
    rank_char = str((idx // 8) + 1)
    return f"{file_char}{rank_char}"

def tokenize_fen(fen_str, max_len=100):
    """Translates the board into the clone's native language"""
    tokens = [VOCAB.get(c, 0) for c in fen_str] # 0 is PAD_IDX
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return torch.tensor([tokens], dtype=torch.long)

def get_clone_move(model, board, device):
    """The brain: Calculates probabilities and filters for legality."""
    model.eval()
    
    # 1. Look at the board
    fen = board.fen()
    input_ids = tokenize_fen(fen).to(device)
    
    with torch.no_grad():
        # 2. Get the raw logits for all 64 squares
        start_logits, end_logits = model(input_ids)
        
        # 3. Convert to probabilities (0.0 to 1.0)
        start_probs = F.softmax(start_logits[0], dim=0).cpu().numpy()
        end_probs = F.softmax(end_logits[0], dim=0).cpu().numpy()
        
    # 4. Generate all physically legal moves for this position
    legal_moves = list(board.legal_moves)
    move_scores = []
    
    # 5. Score every legal move using your cloned psychology
    for move in legal_moves:
        uci = move.uci()
        # Handle the promotion edge-case we discussed
        is_promotion = len(uci) == 5 
        
        start_sq = uci[:2]
        end_sq = uci[2:4]
        
        # We need our square_to_index function from the training script
        s_idx = square_to_index(start_sq)
        e_idx = square_to_index(end_sq)
        
        # The joint probability: P(Start) * P(End)
        joint_prob = start_probs[s_idx] * end_probs[e_idx]
        
        # If it's a promotion, we only consider it if it's a Queen (our rule)
        if is_promotion and uci[4] != 'q':
            joint_prob = 0.0
            
        move_scores.append((joint_prob, move))
        
    # 6. Sort moves by confidence (highest first)
    move_scores.sort(key=lambda x: x[0], reverse=True)
    
    best_move = move_scores[0][1]
    confidence = move_scores[0][0] * 100
    
    return best_move, confidence

def play_terminal_game():
    """Starts a live game against your clone."""
    device = torch.device("cpu") # Inference is light enough for CPU
    
    # Load the trained brain
    print("Waking up the Brahim Clone...")
    model = BrahimClone().to(device)
    model.load_state_dict(torch.load("brahim_clone_10min.pth", map_location=device))
    
    board = chess.Board()
    
    print("\n--- The Mirror Match Begins ---")
    print("You are White. The Clone is Black.")
    
    while not board.is_game_over():
        print(f"\n{board}")
        
        if board.turn == chess.WHITE:
            user_move = input("\nEnter your move (e.g. e2e4): ")
            try:
                board.push_uci(user_move)
            except ValueError:
                print("Invalid or illegal move. Try again.")
        else:
            print("\nClone is thinking...")
            time.sleep(1) # Dramatic effect
            
            clone_move, confidence = get_clone_move(model, board, device)
            
            print(f"Clone plays: {clone_move} (Confidence: {confidence:.4f}%)")
            board.push(clone_move)
            
    print(f"\nGame Over! Result: {board.result()}")

# Start the game
if __name__ == "__main__":
    play_terminal_game()