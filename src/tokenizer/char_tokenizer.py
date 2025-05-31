from typing import List

class CharTokenizer:
    def __init__(self, text:str):
        # get all unique characters from the input text
        chars = sorted(set(text))

        # special token for unknown characters
        chars.append("<unk>")

        # build mappings
        self.stoi = {ch: i  for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s:str) -> List[int]:
        # convert a string into a list of token IDs
        unk_id = self.stoi["<unk>"]
        return [self.stoi.get(c, unk_id) for c in s]

    def decode(self, indices: List[int]) -> str:
        # convert a lsit of token IDs into a string
        return ''.join(self.itos[i] for i in indices)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)