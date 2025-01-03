import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.sparse_ae import SparseAutoencoder, l1_sparsity
import argparse
import os


def train_sparse_autoencoder(
    embeddings,
    encoding_dim=128,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    sparsity_strength=0.001,
    output_path="sparse_ae.pth",
):
    input_dim = embeddings.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseAutoencoder(input_dim, encoding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            decoded, encoded = model(data)
            reconstruction_loss = criterion(decoded, data)
            sparsity_loss = l1_sparsity(encoded, model.sparsity_level)
            loss = reconstruction_loss + sparsity_strength * sparsity_loss
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

    torch.save(model.state_dict(), output_path)
    print(f"Trained sparse autoencoder saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder")
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to the LLM embeddings file (.pt)",
    )
    parser.add_argument(
        "--encoding_dim",
        type=int,
        default=128,
        help="Dimensionality of the latent space",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--sparsity_strength",
        type=float,
        default=0.001,
        help="Strength of the sparsity penalty",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="sparse_ae.pth",
        help="Path to save the trained model",
    )

    args = parser.parse_args()

    if not os.path.exists(args.embeddings_path):
        print(f"Error: Embeddings file not found at {args.embeddings_path}")
        exit()

    embeddings = torch.load(args.embeddings_path)
    train_sparse_autoencoder(
        embeddings,
        args.encoding_dim,
        args.epochs,
        args.batch_size,
        args.lr,
        args.sparsity_strength,
        args.output_path,
    )
