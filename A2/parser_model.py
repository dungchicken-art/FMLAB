import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParserModel(nn.Module):
    def __init__(self, embeddings, n_features=36,
                 hidden_size=200, n_classes=3, dropout_prob=0.5):
        super(ParserModel, self).__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size

        # embedding ma trận (đông cứng, không học)
        self.embeddings = nn.Parameter(
            torch.tensor(embeddings, dtype=torch.float32), requires_grad=False
        )

        # Linear: (n_features * embed_size) → hidden_size
        self.embed_to_hidden_weight = nn.Parameter(
            torch.empty(self.n_features * self.embed_size, self.hidden_size)
        )
        nn.init.xavier_uniform_(self.embed_to_hidden_weight)

        self.embed_to_hidden_bias = nn.Parameter(torch.empty(self.hidden_size))
        nn.init.uniform_(self.embed_to_hidden_bias)

        # dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Linear: hidden_size → n_classes
        self.hidden_to_logits_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.n_classes)
        )
        nn.init.xavier_uniform_(self.hidden_to_logits_weight)

        self.hidden_to_logits_bias = nn.Parameter(torch.empty(self.n_classes))
        nn.init.uniform_(self.hidden_to_logits_bias)

    def embedding_lookup(self, w):
        """
        w: tensor có shape (batch_size, n_features)
        Trả về tensor (batch_size, n_features * embed_size)
        """
        x = self.embeddings[w]  # (batch_size, n_features, embed_size)
        x = x.reshape(x.shape[0], -1)  # Flatten
        return x

    def forward(self, w):
        """
        Forward pass của mô hình
        """
        x = self.embedding_lookup(w)
        h = torch.matmul(x, self.embed_to_hidden_weight) + self.embed_to_hidden_bias
        h = F.relu(h)
        h = self.dropout(h)
        logits = torch.matmul(h, self.hidden_to_logits_weight) + self.hidden_to_logits_bias
        return logits


# ================== KIỂM TRA ===============
def check_embedding(model):
    inds = torch.randint(0, 100, (4, 36), dtype=torch.long)
    selected = model.embedding_lookup(inds)
    assert torch.all(selected == 0), (
        f"The result of embedding lookup: {selected} contains non-zero elements."
    )
    print("Embedding_lookup sanity check passes!")


def check_forward(model):
    inputs = torch.randint(0, 100, (4, 36), dtype=torch.long)
    out = model(inputs)
    expected_out_shape = (4, 3)
    assert out.shape == expected_out_shape, (
        f"The result shape of forward is: {out.shape}, expected {expected_out_shape}"
    )
    print("Forward sanity check passes!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity checks for ParserModel")
    parser.add_argument("-e", "--embedding", action="store_true", help="Check embedding lookup")
    parser.add_argument("-f", "--forward", action="store_true", help="Check forward pass")
    args = parser.parse_args()

    embeddings = np.zeros((100, 30), dtype=np.float32)
    model = ParserModel(embeddings)

    if args.embedding:
        check_embedding(model)
    if args.forward:
        check_forward(model)
