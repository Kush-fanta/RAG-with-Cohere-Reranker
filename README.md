# RAG with Cohere Reranker

This project is a web-based application built with Streamlit that implements a full Retrieval-Augmented Generation (RAG) pipeline. Users can upload a PDF document, process and store it in a vector database, and then ask questions about the document's content. The system uses a sophisticated reranker to improve the relevance of retrieved context before generating a final answer with a large language model.

---
## Key Features

- **Dynamic Document Upload:** Upload any PDF document for analysis.
- **Vector Embeddings:** Automatically chunks the document, creates embeddings, and stores them in a Pinecone vector index.
- **Advanced Retrieval:** Uses a two-stage retrieval process:
    1.  **Semantic Search:** Fetches an initial set of relevant document chunks from Pinecone.
    2.  **Reranking:** Employs Cohere's Rerank model to refine and prioritize the most relevant chunks.
- **Grounded Q&A:** Generates answers using a large language model (LLM) strictly based on the retrieved and reranked context.
- **Chat Interface:** Interact with your document through a familiar chatbot-style interface that stores chat history.
- **Source Verification:** Displays the exact source chunks used to generate each answer for transparency and fact-checking.
- **Automatic Index Creation:** Automatically creates the necessary Pinecone index if it doesn't already exist.

---
## Technology Stack

- **Application Framework:** Streamlit
- **Vector Database:** Pinecone (Serverless)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (from Hugging Face)
- **Reranker:** Cohere (`rerank-english-v3.0`)
- **LLM:** Groq (`llama-3.1-8b-instant`)
- **Core Libraries:** LangChain, PyPDFLoader, python-dotenv

---
## Architecture

The application follows a standard RAG pipeline orchestrated by LangChain and served via a Streamlit frontend.

1.  **Frontend (Streamlit):** The user uploads a PDF and enters API keys.
2.  **Indexing:**
    - The PDF is loaded and split into smaller, overlapping text chunks.
    - The `HuggingFaceEmbeddings` model converts each chunk into a 384-dimension vector.
    - These vectors are "upserted" into a serverless Pinecone index for storage.
3.  **Retrieval & Answering:**
    - A user's query is converted into a vector.
    - Pinecone performs a semantic search to retrieve the top 10 most similar document chunks.
    - The Cohere Rerank model re-orders these 10 chunks, returning the top 3 most relevant ones.
    - The reranked chunks are passed as context to the Groq LLM.
    - The LLM generates an answer based *only* on the provided context.
4.  **Display:** The final answer and its sources are displayed in the Streamlit chat interface.

---
## Setup and Installation

Follow these steps to run the project locally.

**1. Clone the Repository:**
```bash
git clone https://github.com/Kush-fanta/RAG-with-Cohere-Reranker
cd RAG-with-Cohere-Reranker
```
**2. Create a Virtual Environment:**
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
```
**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

## How to Use

1.  **Launch the App:**

    ```bash
    streamlit run main.py
    ```

2.  **Enter API Keys:** In the sidebar, enter your API keys for Pinecone, Hugging Face, Groq, and Cohere.

3.  **Upload a PDF:** Use the file uploader to select a PDF document.

4.  **Create Embeddings:** Click the "Create Vector Embeddings" button to process and index the document.

5.  **Ask Questions:** Once the embeddings are created, use the chat input at the bottom of the page to ask questions about your document.

-----

## Vector Index Configuration

The vector database for this project is hosted on Pinecone. The index is configured with the following parameters:

  - **Provider:** Pinecone
  - **Index Name:** `rag-reranker`
  - **Vector Dimensionality:** 384
  - **Similarity Metric:** `cosine`
  - **Cloud Environment:** `aws`
  - **Region:** `us-east-1`
  - **Index Type:** Serverless
  - **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`

-----


A minimal evaluation was performed using a 5-page sample annual report for the fictional company "InnovateSphere Dynamics."

  - **Success Rate:** The system achieved a **Success Rate of 100%**, correctly answering all 5 out of 5 questions.
  - **Precision and Recall:** The system's **precision was flawless**, with all information being factually correct and directly from the source text. The system also showed perfect **recall**, successfully retrieving all parts of multi-part questions and providing comprehensive summaries.
  - **Conclusion:** The application is highly effective for a range of tasks, including fact retrieval, summarization, and handling negative cases where information is not present in the document.

You can check this out on - https://rag-with-cohere-reranker.streamlit.app/
