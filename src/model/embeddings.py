import cupy as cp

class Embeddings:
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Initialize token and positional embeddings + scale factor
        self.token_embedding = cp.random.randn(vocab_size, d_model) * 0.01
        self.pos_embedding = cp.random.randn(max_seq_len, d_model) * 0.01

        self.scale = 1 / cp.sqrt(d_model)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        x: (batch_size, seq_len) of token indices
        Returns: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = x.shape
        token_emb = self.token_embedding[x]                         # (B, T, D)
        pos_emb = self.pos_embedding[:seq_len][cp.newaxis, :, :]    # (1, T, D)
        return self.scale * token_emb + pos_emb