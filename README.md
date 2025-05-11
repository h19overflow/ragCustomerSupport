# RAG-Powered Customer Support Chatbot with Vertex AI Gemini

## Project Overview

This project demonstrates the implementation of a Retrieval Augmented Generation (RAG) system to build an intelligent customer support chatbot. The chatbot leverages Google Cloud's Vertex AI platform, specifically using a Gemini Large Language Model (LLM) for natural language understanding and generation, and a Google text embedding model for semantic search.

The core idea is to enhance the LLM's responses by grounding them in a specific knowledge base derived from a customer support dataset. Instead of relying solely on the LLM's general pre-trained knowledge, the RAG system first retrieves relevant information from this dataset and then uses that information as context for the LLM to generate a more accurate, relevant, and factual answer.

The Bitext General AI Chatbot Customer Support Dataset (available on Kaggle) is used as the knowledge base, where "response" texts serve as the information sources and "instruction" texts serve as sample user queries for testing.

## Features

*   **Data Ingestion & Preprocessing:** Loads and cleans the Bitext customer support dataset.
*   **Knowledge Base Creation:**
    *   Converts customer support "responses" into text embeddings using a Vertex AI text embedding model (e.g., `textembedding-gecko` or `text-embedding-004`).
    *   Stores these embeddings in a FAISS vector index for efficient similarity search.
*   **Retrieval Mechanism:**
    *   Embeds incoming user queries.
    *   Performs a similarity search against the FAISS index to retrieve the most relevant text chunks (customer support responses) from the knowledge base.
*   **Augmented Generation:**
    *   Constructs a detailed prompt for a Vertex AI Gemini model (e.g., `gemini-1.5-flash-001` or `gemini-1.0-pro-001`).
    *   This prompt includes the original user query and the retrieved relevant text chunks as context.
    *   The Gemini model generates a response based on both the query and the provided context.
*   **Comparison:** Includes a mechanism to compare the RAG-generated answer with an answer from the Gemini model without any RAG context, highlighting the benefits of retrieval augmentation.
*   **Modular Design:** The code is structured into logical cells/functions for each step of the RAG pipeline.

## How RAG Works Here

1.  **Indexing (Offline Process):**
    *   The "response" sections from the Bitext dataset are processed.
    *   Each response is converted into a numerical vector (embedding) that captures its semantic meaning.
    *   These embeddings are stored in a FAISS vector index, creating a searchable knowledge base.
2.  **Querying (Online Process):**
    *   A user asks a question (an "instruction" from the dataset for testing).
    *   The user's question is also converted into an embedding.
    *   This query embedding is used to search the FAISS index for the `k` most similar response embeddings.
    *   The original text of these `k` most similar responses is retrieved.
3.  **Generation:**
    *   The original user question and the retrieved response texts (context) are combined into a prompt.
    *   This augmented prompt is sent to a Gemini LLM.
    *   Gemini uses the provided context to generate a factual and relevant answer.

## Technology Stack

*   **Programming Language:** Python 3
*   **Cloud Platform:** Google Cloud Platform (GCP)
*   **AI Services:**
    *   Vertex AI: For accessing Gemini models and embedding models.
    *   Gemini LLM (e.g., `gemini-1.5-flash-001`, `gemini-1.0-pro-001`): For natural language generation.
    *   Vertex AI Text Embedding Model (e.g., `textembedding-gecko@003`, `text-embedding-004`): For creating text embeddings.
*   **Vector Database:** FAISS (for local, in-memory similarity search)
*   **Core Python Libraries:**
    *   `google-cloud-aiplatform`: Vertex AI SDK for Python.
    *   `pandas`: For data manipulation.
    *   `numpy`: For numerical operations.
    *   `faiss-cpu` (or `faiss-gpu`): For vector indexing and search.
    *   `tqdm`: For progress bars.
    *   `scikit-learn` (implicitly used by FAISS, or for potential future evaluation metrics).

## Prerequisites

1.  **Google Cloud Platform (GCP) Account:**
    *   A GCP project with billing enabled.
    *   The Vertex AI API enabled in your project.
2.  **Authentication:**
    *   Google Cloud SDK installed and initialized (`gcloud init`).
    *   Application Default Credentials set up (e.g., by running `gcloud auth application-default login`).
3.  **Python Environment:**
    *   Python 3.8 or higher recommended.
    *   A virtual environment (e.g., `venv`, `conda`) is highly recommended.
4.  **Dataset:**
    *   Download the "Bitext Sample Customer Support Training Dataset" from Kaggle: [https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset](https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset)
    *   Place the CSV file (e.g., `Bitext_Sample_Customer_Support_Training_Dataset_26k.csv`) in the project directory or update the `DATASET_PATH` variable in the script.

## Installation

1.  **Clone the repository (if applicable) or create a project directory.**
2.  **Set up a Python virtual environment:**
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows: myenv\Scripts\activate
    ```
3.  **Install required Python libraries:**
    ```bash
    pip install --upgrade google-cloud-aiplatform pandas faiss-cpu scikit-learn numpy tqdm
    ```
    *(Note: If you have a compatible GPU and CUDA setup, you can install `faiss-gpu` instead of `faiss-cpu` for potentially faster indexing and search.)*

## Configuration

Before running the script, update the following configuration variables at the beginning of the main Python file (e.g., `rag_pipeline.ipynb` or `rag_pipeline.py`):

*   `GCP_PROJECT_ID`: Your Google Cloud Project ID.
*   `GCP_REGION`: The GCP region where you want to run Vertex AI services (e.g., "us-central1", "asia-southeast1").
*   `DATASET_PATH`: The path to your downloaded Bitext CSV dataset file.
*   `EMBEDDING_MODEL_NAME`: The identifier for the Vertex AI text embedding model (e.g., "textembedding-gecko@003", "text-embedding-004").
*   `TOP_K_RETRIEVAL`: The number of relevant chunks to retrieve for each query.

## Usage

The project is designed to be run as a Python script or a Jupyter Notebook. The code is typically structured in cells that perform sequential steps:

1.  **Setup & Configuration:** Import libraries and set configuration variables.
2.  **Initialize Vertex AI:** Connect to your GCP project.
3.  **Load and Preprocess Data:** Load the Bitext dataset.
4.  **Initialize AI Models:** Load the embedding and generative models from Vertex AI.
5.  **Create Embeddings and Index:** Generate embeddings for the knowledge base and build the FAISS index (this might take some time depending on the dataset size).
6.  **Implement Retrieval Mechanism:** Define the function to retrieve relevant chunks.
7.  **Integrate with Gemini for Generation:** Define the function to generate responses using retrieved context.
8.  **Run Full RAG Pipeline and Test:** Execute the RAG process with sample queries from the dataset.
9.  **(Optional) Compare RAG vs. General LLM:** See the difference in answers with and without RAG.

Run the cells/script sequentially. The output will show the processed queries, retrieved chunks, and the final generated answers from the RAG system, as well as comparisons if that cell is included.

## Potential Future Enhancements


*   **Advanced Chunking Strategies:** Implement sentence splitting, fixed-size chunking with overlap, or semantic chunking for better retrieval from long documents.
*   **Alternative Vector Databases:** Integrate with managed vector databases like Vertex AI Vector Search, Pinecone, or Weaviate for scalability and production use.
*   **Metadata Filtering:** Utilize 'intent' and 'category' from the dataset to filter search results in the vector database.
*   **Re-ranking:** Implement a re-ranking step after initial retrieval to further refine the context provided to the LLM.
*   **Prompt Engineering:** Experiment extensively with different prompt structures to optimize LLM output.
*   **Evaluation Framework:** Develop a robust evaluation pipeline using metrics like ROUGE, BLEU, MRR, Hit Rate, and human evaluation for both retrieval and generation quality.
*   **Fine-tuning:**
    *   Fine-tune the embedding model for domain-specific relevance.
    *   Fine-tune the generative LLM for better context utilization and response style.
*   **Streaming Responses:** For a more interactive chatbot experience.
*   **User Interface:** Build a simple UI (e.g., using Streamlit or Flask) to interact with the RAG chatbot.
*   **Error Handling & Logging:** Implement more robust error handling and logging.

## Troubleshooting Common Issues

*   **Authentication Errors (`Unable to authenticate your request`):**
    *   Ensure `gcloud auth application-default login` has been run successfully.
    *   Restart your Python kernel/environment after authenticating.
    *   Verify `GCP_PROJECT_ID` is correct.
*   **`'gcloud' is not recognized...`:**
    *   Install the Google Cloud SDK and ensure it's added to your system's PATH.
    *   Close and reopen terminal/command prompt windows.
*   **`404 Publisher Model ... was not found`:**
    *   Double-check the exact model name (e.g., `gemini-1.5-flash-001`, `textembedding-gecko@003`) against the official Google Cloud documentation for Vertex AI models.
    *   Ensure the model is available in the `GCP_REGION` you've selected for your project.
    *   Check if the model requires specific enablement or terms to be accepted in the Vertex AI Model Garden for your project.
    *   Verify Vertex AI API quotas.
*   **Quota Errors:**
    *   Check your Vertex AI API quotas in the GCP console.
    *   Reduce batch sizes or add delays when making many API calls (e.g., during embedding generation).

## Contributing

(Add contribution guidelines if this were an open project, e.g., pull request process, coding standards.)

## License

(Specify a license if applicable, e.g., MIT, Apache 2.0.)
