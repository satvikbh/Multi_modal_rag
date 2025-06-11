import streamlit as st
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
import time
import tempfile

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Validate API key
if not api_key:
    st.error("GEMINI_API_KEY not found in .env file. Please add a valid API key.")
    st.stop()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}. Please check your API key in the .env file.")
    st.stop()

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.title("Multi-PDF RAG Application")
uploaded_files = st.file_uploader("Upload up to 4 PDF documents", type="pdf", accept_multiple_files=True)
if uploaded_files and len(uploaded_files) > 4:
    st.error("Please upload no more than 4 PDF files.")
    uploaded_files = uploaded_files[:4]  # Limit to 4 files
query = st.text_input("Enter your question")
ask_button = st.button("Ask")

# Function to process multiple PDFs and create vector store
@st.cache_resource
def process_documents(_files):
    all_chunks = []
    
    # Process each uploaded file
    for i, file in enumerate(_files):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        
        # Load and split document
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
    
    # Create Chroma vector store from all chunks
    vector_store = Chroma.from_documents(all_chunks, embedding_model, persist_directory="./chroma_db")
    return vector_store

# Function to query Gemini
def query_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error querying Gemini: {e}")
        return "Failed to generate response."

if uploaded_files and ask_button and query:
    start_time = time.time()
    with st.spinner("Processing documents..."):
        vector_store = process_documents(uploaded_files)
    
    with st.spinner("Retrieving relevant chunks..."):
        # Retrieve top 3 relevant chunks from all documents
        docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
    
    # Create prompt for Gemini
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    
    with st.spinner("Generating answer..."):
        answer = query_gemini(prompt)
        end_time = time.time()
    
    st.markdown("**Answer (Multi-PDF RAG):**")
    st.write(answer)
    st.write(f"**Response Time:** {end_time - start_time:.2f} seconds")