from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LLMResponseGenerator:
    def __init__(
        self, model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

    def generate_response(self, question, context):
        """Generates a response based on the question and retrieved context."""
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=500,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id,
            )  # Example generation parameters
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response


if __name__ == "__main__":
    # Example usage
    response_generator = LLMResponseGenerator()
    question = "What is the function of the hippocampus?"
    context = "The hippocampus is a major component of the brain, located in the medial temporal lobe. It plays a crucial role in learning and memory."
    response = response_generator.generate_response(question, context)
    print("Generated Response:", response)
