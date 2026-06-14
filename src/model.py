import torch
import torch.nn as nn


class TextCNN(nn.Module):
    """
    TextCNN: Convolutional Neural Network for text classification.
    Based on Kim (2014) "Convolutional Neural Networks for Sentence Classification".

    Architecture:
        Embedding -> Parallel Conv1D (multiple kernel sizes) -> ReLU ->
        Max-over-time Pooling -> Concatenation -> Dropout -> FC -> Logit
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        num_filters: int = 128,
        kernel_sizes: list = None,
        dropout: float = 0.5,
        num_classes: int = 1,
        pretrained_embeddings=None,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4, 5]

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(pretrained_embeddings, dtype=torch.float32)
            )

        # Parallel Conv1D layers, one per kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch, embed_dim, seq_len)

        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(embedded))    # (batch, num_filters, seq_len - k + 1)
            c = c.max(dim=2)[0]               # (batch, num_filters)
            pooled.append(c)

        h = torch.cat(pooled, dim=1)          # (batch, num_filters * len(kernel_sizes))
        h = self.dropout(h)
        logits = self.fc(h)                   # (batch, num_classes)

        if logits.shape[1] == 1:
            logits = logits.squeeze(1)        # (batch,)

        return logits
