import os
import streamlit as st
import tempfile
import cohere
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# load_dotenv()

# Pinecone configuration
PINECONE_INDEX_NAME = "rag-reranker"
PINECONE_DIMENSION = 384 #Dimension for the "all-MiniLM-L6-v2" model

#Streamlit configuration
st.set_page_config(
    page_title="RAG with Reranker",
    layout="wide"
)

st.title("RAG with Cohere Reranker")
st.markdown("Upload a PDF, create vector embeddings, and ask questions about the document!")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get API keys

st.sidebar.header("API Configuration")
pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
huggingface_token = st.sidebar.text_input("HuggingFace API Token", type="password")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
cohere_api_key = st.sidebar.text_input("Cohere API Key", type="password")

api_keys_valid = all([pinecone_api_key, huggingface_token, groq_api_key, cohere_api_key])

if not api_keys_valid:
    st.warning("Please enter all API keys in the sidebar to continue.")
    st.stop()

# Debug: Show which keys are loaded (without showing the actual keys)
st.sidebar.markdown("### API Keys Status")
st.sidebar.write(f"Pinecone: {'✓' if pinecone_api_key else '✗'}")
st.sidebar.write(f"HuggingFace: {'✓' if huggingface_token else '✗'}")
st.sidebar.write(f"Groq: {'✓' if groq_api_key else '✗'}")
st.sidebar.write(f"Cohere: {'✓' if cohere_api_key else '✗'}")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_token
os.environ["PINECONE_API_KEY"] = pinecone_api_key  # Add this line

#Initialize clients directly without caching
try:
    llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {'device': 'cpu'}
    )
    cohere_client = cohere.Client(api_key=cohere_api_key)
    pinecone_client = Pinecone(api_key=pinecone_api_key)
except Exception as e:
    st.error(f"Failed to initialize clients: {str(e)}")
    llm, embeddings, cohere_client, pinecone_client = None, None, None, None

if None in [llm, embeddings, cohere_client, pinecone_client]:
    st.error("Failed to initialize API clients. Please check your API keys.")
    st.stop()

# check and create new pinecone index
try:
    if PINECONE_INDEX_NAME not in pinecone_client.list_indexes().names():
        st.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pinecone_client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
except Exception as e:
    st.error(f"Failed to check or create Pinecone index: {e}")
    st.stop()



# document Upload
st.header("Document Upload")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf",
    help="Upload a PDF document to create vector embeddings"
)

# vector Embedding Creation
if uploaded_file is not None:
    st.info(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")

    if st.button("Create Vector Embeddings", type="primary"):
        with st.spinner("Creating vector embeddings... This may take a few minutes."):
            try:
                # Debug: Check if we have the API key
                st.write(f"DEBUG: Pinecone API Key exists: {bool(pinecone_api_key)}")
                st.write(f"DEBUG: Pinecone API Key length: {len(pinecone_api_key) if pinecone_api_key else 0}")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
                docs = text_splitter.split_documents(documents)

                # Create vector store with explicit index reference
                vector_store = PineconeVectorStore.from_documents(
                    documents=docs,
                    embedding=embeddings,
                    index_name=PINECONE_INDEX_NAME
                )

                base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
                reranker = CohereRerank(
                    client=cohere_client,
                    model="rerank-english-v3.0",
                    top_n=3
                )
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=reranker,
                    base_retriever=base_retriever
                )

                # RAG Chain - 
                prompt = ChatPromptTemplate.from_template("""
                Answer the user's question based only on the following context.
                <context>
                {context}
                </context>
                Answer the below question.
                Question: {input}
                """)

                document_chain = create_stuff_documents_chain(llm, prompt)
                retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)

                #Store in session state
                st.session_state.vector_store = vector_store
                st.session_state.retrieval_chain = retrieval_chain
                st.session_state.messages = [] # Clear previous chat history

                os.unlink(tmp_file_path)

                st.success(f"Successfully processed {len(docs)} document chunks!")

            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

#Chat Interface
st.header("Chat about the Document")

if st.session_state.retrieval_chain is not None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your document"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching for answer..."):
                try:
                    response = st.session_state.retrieval_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    st.markdown(answer)

                    with st.expander("Show Sources"):
                        for i, doc in enumerate(response["context"]):
                            st.text_area(
                                f"Source {i+1}:",
                                doc.page_content,
                                height=100,
                                key=f"source_{len(st.session_state.messages)}_{i}"
                            )

                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

else:
    st.info("Please upload a PDF and create the vector embeddings to start chatting.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Configuration")
st.sidebar.markdown(f"**Index Name:** {PINECONE_INDEX_NAME}")
st.sidebar.markdown("**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2")
st.sidebar.markdown("**Reranker:** Cohere rerank-english-v3.0")
st.sidebar.markdown("**LLM:** Llama-3.1-8b-instant")

with st.expander("How to use this app"):
    st.markdown("""
    1. **Enter API Keys**: Add your API keys in the sidebar
    2. **Upload PDF**: Choose a PDF file to analyze
    3. **Create Embeddings**: Click the button to process your document
    4. **Ask Questions**: Type questions about your document
    5. **View Results**: See answers with source citations
    """)

