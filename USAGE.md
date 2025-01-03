# Usage

This document provides detailed instructions on how to use the code in this repository to perform latent space retrieval for enhanced RAG in computational neuroscience.

## Setting Up Your Environment

Before running any code, ensure you have set up your environment correctly:

1. **Prerequisites:** Verify that you have the necessary software installed:

    - Python 3.8 or higher
    - PyTorch (Installation instructions: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/))
    - Hugging Face Transformers (Installation: `pip install transformers`)
    - Faiss (CPU or GPU version, installation instructions may vary. Refer to the official Faiss repository: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss))
    - NumPy (Installation: `pip install numpy`)
    - Scikit-learn (Installation: `pip install scikit-learn`)
    - Pandas (Installation: `pip install pandas`)

## Preparing Your Data

You'll need to prepare your data for two main stages: generating embeddings and evaluating the RAG system.

### 1. Data for Embedding Generation

The `embedding/embedding_generator.py` script expects a plain text file where each line or block of text represents a chunk of your computational neuroscience literature.

-   **Format:** Plain text (`.txt`)
-   **Structure:** Each line or paragraph should be a meaningful unit of text that you want to embed and later retrieve.
-   **Example (`data/my_neuroscience_data.txt`):**
    ```
    Neurons are the fundamental units of the nervous system.
    Synaptic plasticity is the ability of synapses to strengthen or weaken over time.
    Computational models are used to simulate brain function.
    The hippocampus plays a crucial role in memory formation.
    ```

### 2. Data for Evaluation

The `evaluation/evaluate_rag.py` script expects a JSON file containing questions, their corresponding relevant context snippets (which will be used as the knowledge base), and the ground truth answers.

-   **Format:** JSON (`.json`)
-   **Structure:** A list of JSON objects, where each object has the following keys:
    -   `context`: The relevant text chunk from your corpus.
    -   `question`: The question you want the RAG system to answer.
    -   `answer`: The ground truth answer to the question.
-   **Example (`data/evaluation_data.json`):**
    ```json
    [
        {
            "context": "NMDA receptors are ligand-gated ion channels that are crucial for synaptic plasticity and learning. They are activated by glutamate and require depolarization of the postsynaptic neuron to remove a magnesium block.",
            "question": "What is the role of NMDA receptors in synaptic plasticity?",
            "answer": "NMDA receptors are crucial for synaptic plasticity and learning."
        },
        {
            "context": "The hippocampus is a brain region located in the medial temporal lobe and is essential for the formation of new episodic memories.",
            "question": "What is the function of the hippocampus?",
            "answer": "The hippocampus is essential for the formation of new episodic memories."
        }
    ]
    ```

## Running the Code

This section explains how to run each of the Python scripts provided in the repository.

### 1. Generating LLM Embeddings (`embedding/embedding_generator.py`)

This script generates embeddings for your text data using a pre-trained LLM.

**Command:**

```bash
python embedding/embedding_generator.py --data_path <path_to_your_data.txt> --output_path <path_to_save_embeddings.pt>
```

### Command-Line Arguments:

-   `--data_path`: (Required) The path to your plain text data file (e.g., `data/my_neuroscience_data.txt`).
-   `--output_path`: (Required) The path where you want to save the generated embeddings as a PyTorch `.pt` file (e.g., `data/embeddings.pt`).
-   `--model_name`: (Optional) The name of the pre-trained Hugging Face Transformers model to use for embedding generation. Defaults to `"sentence-transformers/all-mpnet-base-v2"`.
-   `--device`: (Optional) The device to use for computation (`cuda` or `cpu`). Defaults to `cuda` if available, otherwise `cpu`.

### Example Usage:

```bash
python embedding/embedding_generator.py --data_path data/my_neuroscience_data.txt --output_path data/embeddings.pt
```
