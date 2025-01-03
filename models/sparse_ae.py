import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, sparsity_level=0.05):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * encoding_dim),
            nn.ReLU(),
            nn.Linear(2 * encoding_dim, encoding_dim),
            nn.Sigmoid(),  # Sigmoid to keep activations between 0 and 1 for easier sparsity control
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 2 * encoding_dim),
            nn.ReLU(),
            nn.Linear(2 * encoding_dim, input_dim),
        )
        self.sparsity_level = sparsity_level

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


def l1_sparsity(encoded, target_sparsity):
    """Calculates the L1 sparsity penalty."""
    return torch.mean(torch.abs(encoded))


if __name__ == "__main__":
    # Example usage and testing
    input_dim = 768  # Example embedding dimension
    encoding_dim = 128
    batch_size = 10
    dummy_input = torch.randn(batch_size, input_dim)

    model = SparseAutoencoder(input_dim, encoding_dim)
    decoded, encoded = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Encoded shape:", encoded.shape)
    print("Decoded shape:", decoded.shape)

    sparsity = l1_sparsity(encoded, model.sparsity_level)
    print("Sparsity:", sparsity.item())
