from transformers import AutoTokenizer, AutoModel
import torch


class EmbeddingGenerator:
    def __init__(
        self,
        model_name="sentence-transformers/all-mpnet-base-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def generate_embeddings(self, texts):
        """Generates embeddings for a list of texts."""
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling to get sentence embeddings (mean pooling in this example)
            embeddings = self.mean_pooling(
                model_output, encoded_input["attention_mask"]
            )
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


if __name__ == "__main__":
    # Example usage
    embedding_generator = EmbeddingGenerator()
    texts = [
        "This is a sentence about neurons.",
        "Another sentence discussing synapses.",
    ]
    embeddings = embedding_generator.generate_embeddings(texts)
    print("Embeddings shape:", embeddings.shape)
    print("Sample embeddings:", embeddings[:2])
