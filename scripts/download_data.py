import requests
import os
def download_shakespeare():
    """Download the Shakespeare dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    # Create data directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)

    print("Downloading Shakespeare dataset...")
    response = requests.get(url)

    with open("data/raw/shakespeare.txt", "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Downloaded {len(response.text)} characters to data/raw/shakespeare.txt")
if __name__ == "main":
    download_shakespeare()