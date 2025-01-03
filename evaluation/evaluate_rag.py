import torch
from embedding.embedding_generator import EmbeddingGenerator
from models.sparse_ae import SparseAutoencoder
from retrieval.latent_retriever import LatentSpaceRetriever
from response_generation.response_generator import LLMResponseGenerator
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import json


def evaluate_rag(data_path, sparse_ae_path, encoding_dim, top_k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data (assuming a simple JSON format for now)
    with open(data_path, "r") as f:
        evaluation_data = json.load(f)

    text_chunks = [item["context"] for item in evaluation_data]
    questions = [item["question"] for item in evaluation_data]
    ground_truth_answers = [item["answer"] for item in evaluation_data]

    # Initialize models
    embedding_generator = EmbeddingGenerator()
    initial_embeddings = embedding_generator.generate_embeddings(text_chunks)

    sparse_ae_model = SparseAutoencoder(
        input_dim=initial_embeddings.shape[1], encoding_dim=encoding_dim
    ).to(device)
    sparse_ae_model.load_state_dict(torch.load(sparse_ae_path, map_location=device))
    sparse_ae_model.eval()

    latent_retriever = LatentSpaceRetriever(
        sparse_ae_model, embedding_generator, text_chunks
    )
    response_generator = LLMResponseGenerator()

    # Evaluation loop
    latent_rag_results = []
    traditional_rag_results = []

    for i, question in enumerate(questions):
        # Latent Space RAG
        latent_retrieved_contexts = latent_retriever.retrieve_context(
            question, top_k=top_k
        )
        latent_response = response_generator.generate_response(
            question, "\n".join(latent_retrieved_contexts)
        )
        latent_rag_results.append(latent_response)

        # Traditional RAG (using embedding distance directly)
        query_embedding = (
            embedding_generator.generate_embeddings([question]).cpu().numpy()
        )
        similarities = cosine_similarity(
            query_embedding, initial_embeddings.cpu().numpy()
        )
        top_indices = similarities.argsort()[0][-top_k:][::-1]
        traditional_retrieved_contexts = [text_chunks[i] for i in top_indices]
        traditional_response = response_generator.generate_response(
            question, "\n".join(traditional_retrieved_contexts)
        )
        traditional_rag_results.append(traditional_response)

        print(f"Evaluated question {i+1}/{len(questions)}")

    # Calculate basic evaluation metrics (this is a placeholder - implement more robust metrics)
    # Example: You could compare the generated responses to the ground truth answers using metrics like BLEU, ROUGE, etc.
    print("\n--- Evaluation Summary ---")
    print("Latent Space RAG Responses (first 5):", latent_rag_results[:5])
    print("Traditional RAG Responses (first 5):", traditional_rag_results[:5])
    print("\nNote: Implement more comprehensive evaluation metrics here.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Latent Space RAG")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the evaluation data file (JSON)",
    )
    parser.add_argument(
        "--sparse_ae_path",
        type=str,
        required=True,
        help="Path to the trained sparse autoencoder model (.pth)",
    )
    parser.add_argument(
        "--encoding_dim",
        type=int,
        default=128,
        help="Dimensionality of the latent space",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of context chunks to retrieve"
    )

    args = parser.parse_args()

    evaluate_rag(args.data_path, args.sparse_ae_path, args.encoding_dim, args.top_k)
