import requests
import time

def scrape_lichess_games(username):
    # Lichess also requires a custom User-Agent. 
    # Providing an email or valid contact prevents them from blocking your IP.
    headers = {
        "Accept": "application/x-chess-pgn",
        "User-Agent": f"BrahimClone-DataScraper (Contact: {username}@lichess.org)"
    }
    
    # We ask for the evaluations, clock times, and opening names to enrich our dataset
    params = {
        "evals": "true",
        "clocks": "true",
        "opening": "true"
    }

    url = f"https://lichess.org/api/games/user/{username}"
    output_filename = f"{username}_lichess_raw_games.pgn"
    
    print(f"Opening Lichess data stream for: {username}...")
    print("This might take a few minutes depending on the number of games. Do not interrupt...")

    try:
        # stream=True is critical here. It keeps the connection open and writes 
        # the data in chunks rather than trying to load 5,000 games into your RAM all at once.
        with requests.get(url, headers=headers, params=params, stream=True, timeout=30) as response:
            response.raise_for_status()
            
            with open(output_filename, "w", encoding="utf-8") as file:
                # Iterate through the incoming stream in 8KB chunks
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: # Filter out keep-alive new chunks
                        file.write(chunk.decode("utf-8"))
                        
        print(f"\nExtraction complete! All Lichess games saved to '{output_filename}'.")
        
    except requests.exceptions.Timeout:
        print("\n[!] The connection timed out. Lichess took too long to stream the data.")
    except requests.exceptions.RequestException as e:
        print(f"\n[!] An error occurred during the stream: {e}")

# Execute the extraction
scrape_lichess_games("brahimkhattara")