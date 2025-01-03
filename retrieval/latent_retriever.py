import faiss
import numpy as np
import torch


class LatentSpaceRetriever:
    def __init__(self, sparse_autoencoder, embedding_generator, text_chunks):
        self.autoencoder = sparse_autoencoder
        self.embedding_generator = embedding_generator
        self.text_chunks = text_chunks
        self.latent_embeddings = self._encode_chunks_to_latent(text_chunks)
        self.index = self._build_faiss_index(self.latent_embeddings)

    def _encode_chunks_to_latent(self, text_chunks):
        """Encodes text chunks into the latent space."""
        with torch.no_grad():
            llm_embeddings = (
                self.embedding_generator.generate_embeddings(text_chunks).cpu().numpy()
            )
            latent_embeddings = (
                self.autoencoder.encoder(torch.tensor(llm_embeddings)).cpu().numpy()
            )
        return latent_embeddings

    def _build_faiss_index(self, latent_embeddings):
        """Builds a FAISS index for efficient similarity search."""
        dimension = latent_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
        index.add(np.array(latent_embeddings).astype("float32"))
        return index

    def retrieve_context(self, query, top_k=5):
        """Retrieves the top_k most relevant context chunks based on latent space similarity."""
        with torch.no_grad():
            query_embedding = (
                self.embedding_generator.generate_embeddings([query]).cpu().numpy()
            )
            query_latent = (
                self.autoencoder.encoder(torch.tensor(query_embedding)).cpu().numpy()
            )

        distances, indices = self.index.search(
            np.array(query_latent).astype("float32"), top_k
        )
        retrieved_contexts = [self.text_chunks[i] for i in indices[0]]
        return retrieved_contexts


if __name__ == "__main__":
    # Example usage (you'll need to have a trained autoencoder and embeddings)
    from embedding.embedding_generator import EmbeddingGenerator
    from models.sparse_ae import SparseAutoencoder

    # Dummy data for testing
    text_chunks = [
        "This is about neurons.",
        "Synapses transmit information.",
        "Computational models of the brain.",
    ]

    # Load a dummy autoencoder (replace with your trained model)
    input_dim = 768
    encoding_dim = 128
    sparse_ae_model = SparseAutoencoder(input_dim, encoding_dim)

    # Load a dummy embedding generator
    embedding_generator = EmbeddingGenerator()

    retriever = LatentSpaceRetriever(sparse_ae_model, embedding_generator, text_chunks)
    query = "Tell me about brain cells."
    relevant_contexts = retriever.retrieve_context(query)
    print("Retrieved Contexts:", relevant_contexts)
