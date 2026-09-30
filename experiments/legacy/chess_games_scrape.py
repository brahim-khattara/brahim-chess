import requests
import time

def scrape_chesscom_games(username):
    headers = {
        "User-Agent": f"BrahimClone-DataScraper (Contact: {username}@chess.com)" 
    }
    
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    print(f"Fetching archive index for: {username}...")
    
    try:
        # Added a 10-second timeout here
        response = requests.get(archives_url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch archives: {e}")
        return

    data = response.json()
    archives = data.get("archives", [])
    
    if not archives:
        print("No games found or invalid username.")
        return
        
    print(f"Found {len(archives)} months of game data. Starting download...")
    output_filename = f"{username}_chess_raw_games.pgn"
    
    # Changed to 'w' (write) so it starts fresh every time you run the script
    with open(output_filename, "w", encoding="utf-8") as file:
        for month_url in archives:
            pgn_url = f"{month_url}/pgn"
            print(f"Downloading: {pgn_url}...")
            
            # The Retry Engine
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # CRITICAL: 15-second timeout. If it hangs, it breaks and retries.
                    pgn_response = requests.get(pgn_url, headers=headers, timeout=15)
                    
                    if pgn_response.status_code == 200:
                        file.write(pgn_response.text)
                        file.write("\n") 
                        break # Success! Break out of the retry loop.
                    else:
                        print(f"  [!] Attempt {attempt+1} failed (HTTP {pgn_response.status_code}). Retrying...")
                        time.sleep(2)
                
                except requests.exceptions.Timeout:
                    print(f"  [!] Attempt {attempt+1} timed out! Server hung. Retrying...")
                    time.sleep(2)
                except requests.exceptions.RequestException as e:
                    print(f"  [!] Attempt {attempt+1} encountered an error: {e}. Retrying...")
                    time.sleep(2)
            else:
                # This only runs if all 3 retries fail
                print(f"  [!!!] Failed to download {pgn_url} after {max_retries} attempts. Skipping month.")
                
            # The rate-limit safety buffer
            time.sleep(1.5)
            
    print(f"\nExtraction complete! Data saved to '{output_filename}'.")

# Execute the extraction
scrape_chesscom_games("brahimkhattara")