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
from PIL import Image
import pytesseract
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename="gemini_response.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

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
st.title("Multimodal RAG Application")
st.write("Upload up to 4 files (PDFs or images, e.g., book pages or scanned documents).")
uploaded_files = st.file_uploader("Upload PDFs or Images", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
if uploaded_files and len(uploaded_files) > 4:
    st.error("Please upload no more than 4 files.")
    uploaded_files = uploaded_files[:4]  # Limit to 4 files

st.write("Enter your question and optionally upload an image (e.g., a specific book page or diagram).")
query_text = st.text_input("Enter your question (e.g., 'Explain the content in the uploaded image')")
query_image = st.file_uploader("Upload an image for the query (optional)", type=["png", "jpg", "jpeg"], key="query_image")
ask_button = st.button("Ask")

# Function to extract text from image using OCR
def extract_text_from_image(image_file):
    try:
        # Open image using PIL
        image = Image.open(image_file)
        # Perform OCR to extract text
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        logging.error(f"Image OCR error: {e}")
        return ""

# Function to process PDFs and images, create vector store
@st.cache_resource
def process_documents(_files):
    all_chunks = []
    
    # Process each uploaded file
    for file in _files:
        file_extension = os.path.splitext(file.name)[1].lower()
        
        if file_extension == '.pdf':
            # Save uploaded PDF file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = tmp_file.name
            
            # Load and split PDF document
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            # Process image file
            image_text = extract_text_from_image(file)
            if image_text:
                st.write(f"**Extracted Text from Image {file.name}:**")
                st.write(image_text[:500] + "..." if len(image_text) > 500 else image_text)
                # Add image text as a chunk for vector store
                all_chunks.append(type('Document', (), {'page_content': image_text, 'metadata': {'type': 'image', 'source': file.name}})())
    
    # Create Chroma vector store from all chunks
    try:
        vector_store = Chroma.from_documents(all_chunks, embedding_model, persist_directory="./chroma_db_multimodal")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Function to query Gemini
def query_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2000,
            )
        )
        # Log the response for debugging
        logging.debug(f"Gemini query response: {response.text}")
        return response.text
    except Exception as e:
        st.error(f"Error querying Gemini: {e}")
        logging.error(f"Query error: {e}")
        return "Failed to generate response."

if uploaded_files and ask_button and query_text:
    start_time = time.time()
    
    with st.spinner("Processing documents and images..."):
        vector_store = process_documents(uploaded_files)
        if vector_store is None:
            st.error("Failed to process inputs. Please try again.")
            st.stop()
    
    with st.spinner("Retrieving relevant chunks and image content..."):
        # Retrieve top 5 relevant chunks
        docs = vector_store.similarity_search(query_text, k=5)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Process query image if provided
        query_image_text = ""
        if query_image:
            query_image_text = extract_text_from_image(query_image)
            if query_image_text:
                st.write("**Extracted Text from Query Image:**")
                st.write(query_image_text[:500] + "..." if len(query_image_text) > 500 else query_image_text)
    
    # Create prompt for Gemini
    prompt = f"""You are an expert at analyzing book content. Based on the following context from book PDFs and images, and text extracted from an uploaded query image (if provided), provide a comprehensive answer to the question. If the question asks to explain content in the image, summarize the book context and use the extracted image text to describe the content.

Book Document and Image Context:
{context}

Extracted Query Image Text (if provided):
{query_image_text if query_image_text else "No query image provided."}

Question: {query_text}

Please provide a detailed answer. If explaining content in an image, summarize the book context and describe the content based on the image text and document context. If the information is not sufficient, mention what additional information might be needed.

Answer:"""
    
    with st.spinner("Generating answer..."):
        answer = query_gemini(prompt)
        end_time = time.time()
    
    # Display results
    st.markdown("**Answer (Multimodal RAG):**")
    st.write(answer)
    st.write(f"**Response Time:** {end_time - start_time:.2f} seconds")
    
    # Display some stats
    with st.expander("Input Statistics"):
        st.write(f"**Retrieved Chunks:** {len(docs)}")
        st.write(f"**Query Image Text Length:** {len(query_image_text)} characters" if query_image_text else "**No Query Image Provided**")

# Clean up temporary files on app restart
for temp_file in [f for f in os.listdir() if f.startswith("tmp") and f.endswith(".pdf")]:
    try:
        os.remove(temp_file)
    except:
        pass