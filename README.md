# RAG-Powered Customer Support Chatbot with Vertex AI Gemini

## Project Overview

This project implements a Retrieval Augmented Generation (RAG) system to create an intelligent customer support chatbot. It utilizes Google Cloud's Vertex AI platform, featuring a Gemini Large Language Model (LLM) for understanding and generating natural language, alongside a Google text embedding model for effective semantic search.

The fundamental goal is to enhance the LLM's responses by grounding them in a specific, factual knowledge base. This knowledge base is derived from the "Bitext General AI Chatbot Customer Support Dataset." Instead of relying solely on the LLM's broad pre-trained knowledge, the RAG system first retrieves pertinent information from this dataset. This retrieved information then serves as context, enabling the LLM to generate answers that are more accurate, relevant, and aligned with the provided customer support scenarios.

In this setup, the "response" texts from the Bitext dataset form the core of our knowledge base, while the "instruction" texts are used as sample user queries to test and demonstrate the system's capabilities.

## Features

*   **Data Ingestion & Preprocessing:** Loads the Bitext customer support dataset and performs necessary cleaning.
*   **Knowledge Base Creation:**
    *   Converts customer support "responses" into numerical text embeddings using a Vertex AI text embedding model (e.g., `textembedding-gecko@003` or `text-embedding-004`).
    *   Stores these embeddings efficiently in a FAISS vector index to enable rapid similarity searches.
*   **Retrieval Mechanism:**
    *   Generates an embedding for incoming user queries.
    *   Queries the FAISS index to find and retrieve the most semantically similar text chunks (customer support responses) from the indexed knowledge base.
*   **Augmented Generation:**
    *   Constructs a comprehensive prompt for a Vertex AI Gemini model (such as `gemini-1.5-flash-001` or `gemini-1.0-pro-001`).
    *   This prompt includes the original user query augmented with the relevant text chunks retrieved in the previous step.
    *   The Gemini model then generates a response based on both the user's query and the specific context provided.
*   **Comparative Analysis:** Includes functionality to compare the RAG-generated answer with an answer from the Gemini model operating without any RAG-provided context, thereby highlighting the benefits of retrieval augmentation.
*   **Modular Design:** The Python code is organized into logical cells (for Jupyter Notebooks) or functions, representing each distinct stage of the RAG pipeline.

## How RAG Works in This Project

1.  **Indexing (Typically an Offline Process):**
    *   The "response" sections (chatbot answers) from the Bitext dataset are processed.
    *   Each response is transformed into a numerical vector, known as an embedding, which captures its semantic meaning.
    *   These embeddings are then loaded into a FAISS vector index. This creates a searchable, meaning-based representation of the knowledge base.
2.  **Querying (Real-time Process):**
    *   A user poses a question (for testing, an "instruction" from the dataset is used).
    *   This user question is also converted into an embedding using the same model.
    *   The query embedding is used to search the FAISS index to find the `k` most similar response embeddings (and thus, the most relevant pieces of knowledge).
    *   The original text of these `k` most similar responses is retrieved.
3.  **Generation:**
    *   The original user question and the retrieved response texts (which act as context) are combined to form a detailed prompt.
    *   This augmented prompt is then sent to a Gemini Large Language Model.
    *   Gemini utilizes the provided context to generate a factual, relevant, and nuanced answer to the user's question.

## Technology Stack

*   **Programming Language:** Python 3
*   **Cloud Platform:** Google Cloud Platform (GCP)
*   **AI Services (Vertex AI):**
    *   Gemini LLM (e.g., `gemini-1.5-flash-001`, `gemini-1.0-pro-001`): For advanced natural language generation.
    *   Vertex AI Text Embedding Model (e.g., `textembedding-gecko@003`, `text-embedding-004`): For creating semantic text embeddings.
*   **Vector Database:** FAISS (for efficient, local, in-memory similarity search)
*   **Core Python Libraries:**
    *   `google-cloud-aiplatform`: The official Vertex AI SDK for Python.
    *   `pandas`: For structured data manipulation.
    *   `numpy`: For numerical computations, especially with vectors.
    *   `faiss-cpu` (or `faiss-gpu`): For building and searching the vector index.
    *   `tqdm`: For displaying progress bars during long operations.

## Prerequisites

1.  **Google Cloud Platform (GCP) Account:**
    *   A configured GCP project with billing enabled.
    *   The **Vertex AI API** must be enabled for this project.
2.  **Authentication with GCP:**
    *   The Google Cloud SDK (gcloud CLI) must be installed and initialized (`gcloud init`).
    *   Application Default Credentials (ADC) should be set up, typically by running `gcloud auth application-default login`. This allows your local script to securely access GCP services.
3.  **Python Environment:**
    *   Python 3.8 or a newer version is recommended.
    *   Using a Python virtual environment (e.g., via `venv` or `conda`) is strongly advised to manage dependencies.
4.  **Dataset:**
    *   The "Bitext Sample Customer Support Training Dataset" from Kaggle is required. It can be downloaded from: [https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset](https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset)
    *   The downloaded CSV file (e.g., `Bitext_Sample_Customer_Support_Training_Dataset_26k.csv` or a similar name) should be placed in the project directory, or the `DATASET_PATH` variable in the script must be updated accordingly.

## Installation

1.  **Set up your project directory.** If cloned from a repository, navigate into it.
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv rag_env
    source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
    ```
3.  **Install the necessary Python libraries:**
    ```bash
    pip install --upgrade google-cloud-aiplatform pandas faiss-cpu numpy tqdm
    ```
    *(Note: For users with a compatible NVIDIA GPU and CUDA setup, `faiss-gpu` can be installed instead of `faiss-cpu` for potentially faster vector operations, though `faiss-cpu` is generally sufficient for datasets of this size.)*

## Configuration in the Script

Before executing the main Python script or Jupyter Notebook, ensure the following configuration variables are correctly set at the beginning of the file:

*   `GCP_PROJECT_ID`: Set this to your specific Google Cloud Project ID (e.g., "raggemini-459500").
*   `GCP_REGION`: Specify the GCP region for Vertex AI services (e.g., "us-central1", "asia-southeast1"). Choose a region where the desired Gemini and embedding models are available.
*   `DATASET_PATH`: The local file path to your downloaded Bitext CSV dataset.
*   `EMBEDDING_MODEL_NAME`: The identifier for the Vertex AI text embedding model to be used (e.g., "textembedding-gecko@003", "text-embedding-004").
*   `TOP_K_RETRIEVAL`: An integer specifying how many relevant chunks should be retrieved from the knowledge base for each query (e.g., 3 or 5).

## Usage

This project is structured for execution as a Python script or within a Jupyter Notebook environment. The code is typically divided into cells or functions that perform the RAG pipeline steps in sequence:

1.  **Setup & Configuration:** Imports necessary libraries and defines key configuration parameters.
2.  **Initialize Vertex AI:** Establishes a connection to your GCP project and specified region.
3.  **Load and Preprocess Data:** Loads the Bitext dataset, cleans it, and prepares it for processing.
4.  **Initialize AI Models:** Loads the chosen text embedding model and the Gemini generative model from Vertex AI.
5.  **Create Embeddings and Index:** Generates vector embeddings for the knowledge base (responses) and constructs the FAISS index. This step can be time-consuming for large datasets.
6.  **Implement Retrieval Mechanism:** Defines the logic for embedding a query and retrieving relevant text chunks.
7.  **Integrate with Gemini for Generation:** Defines the logic for constructing an augmented prompt and generating a response using Gemini.
8.  **Run Full RAG Pipeline and Test:** Executes the complete RAG process using sample queries from the dataset to demonstrate functionality.
9.  **(Optional) Compare RAG vs. General LLM:** Illustrates the difference in response quality when using RAG versus relying on the LLM's general knowledge.

Execute the script or notebook cells sequentially. The output will display details of the process, including processed queries, the text of retrieved chunks, and the final answers generated by the RAG system.

## Potential Future Enhancements

*   **Advanced Text Chunking:** Explore techniques like sentence splitting, fixed-size chunking with overlap, or semantic-aware chunking to optimize information retrieval from longer documents.
*   **Managed Vector Databases:** For production scenarios, migrate from local FAISS to scalable, managed vector databases like Vertex AI Vector Search, Pinecone, or Weaviate.
*   **Metadata-Enhanced Retrieval:** Leverage the 'intent' and 'category' fields from the dataset to implement metadata filtering during the search process in the vector database.
*   **Re-ranking Layer:** Introduce a re-ranking step after the initial retrieval phase to further refine the relevance of context chunks provided to the LLM.
*   **Sophisticated Prompt Engineering:** Conduct extensive experimentation with various prompt structures and instructions to fine-tune the LLM's output for desired tone, style, and accuracy.
*   **Comprehensive Evaluation Framework:** Develop a robust framework for evaluating the RAG system, employing metrics such as ROUGE, BLEU, MRR, Hit Rate, and incorporating human evaluation for nuanced aspects of retrieval and generation quality.
*   **Model Fine-tuning (Advanced):**
    *   Fine-tune the embedding model on domain-specific data to improve the understanding of relevance.
    *   Fine-tune the generative LLM to enhance its ability to utilize provided context and adhere to specific response styles.
*   **Streaming for Interactivity:** Implement streaming capabilities for LLM responses to create a more dynamic and interactive chatbot experience.
*   **User Interface:** Develop a simple graphical user interface (e.g., using Streamlit or Flask) to allow easier interaction with the RAG-powered chatbot.
*   **Production-Grade Error Handling & Logging:** Implement more comprehensive error handling mechanisms and detailed logging for monitoring and debugging.

## Troubleshooting Common Issues

*   **Authentication Errors (e.g., `Unable to authenticate your request`):**
    *   Confirm that `gcloud auth application-default login` has been executed successfully in your terminal.
    *   Restart your Python kernel or development environment after authenticating.
    *   Double-check that the `GCP_PROJECT_ID` in your script is correct.
*   **`'gcloud' is not recognized as an internal or external command...`:**
    *   Ensure the Google Cloud SDK is installed on your system.
    *   Verify that the SDK's `bin` directory is included in your system's PATH environment variable.
    *   Close and reopen any terminal or command prompt windows after installation or PATH modification.
*   **`404 Publisher Model ... was not found` Errors:**
    *   Carefully verify the exact model name (e.g., `gemini-1.5-flash-001`, `textembedding-gecko@003`) against the official Google Cloud documentation for available Vertex AI models.
    *   Confirm that the specified model is available in the `GCP_REGION` selected in your script and is accessible to your project.
    *   Check the Vertex AI Model Garden in the GCP Console for any specific enablement steps or terms that might need to be accepted for the model in your project.
    *   Review relevant Vertex AI API quotas for your project.
*   **Quota Exceeded Errors:**
    *   Examine your Vertex AI API quotas in the GCP Console.
    *   If making numerous API calls (e.g., during batch embedding generation), consider reducing batch sizes or introducing delays between calls.
