from typing import List, Tuple
import numpy as np

class CharDataset:
    def __init__(self, token_ids: List[int], seq_len: int, batch_size: int):
        self.data = token_ids
        self.seq_len = seq_len
        self.batch_size = batch_size

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        # Return batch of shape (batch_size, seq_len)
        inputs = []
        targets = []
        for _ in range(self.batch_size):
            start = np.random.randint(0, len(self.data) - self.seq_len - 1)
            x = self.data[start : start + self.seq_len]
            y = self.data[start + 1 : start + self.seq_len + 1]
            inputs.append(x)
            targets.append(y)
        return np.array(inputs), np.array(targets)