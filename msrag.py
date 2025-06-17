import streamlit as st
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import time
import tempfile
from pdf2image import convert_from_path
from PIL import Image
import io
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
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Streamlit UI
st.title("Multimodal Vector RAG Application")
uploaded_files = st.file_uploader("Upload up to 4 PDF documents or images", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
if uploaded_files and len(uploaded_files) > 4:
    st.error("Please upload no more than 4 files (PDFs or images).")
    uploaded_files = uploaded_files[:4]  # Limit to 4 files
query = st.text_input("Enter your question")
ask_button = st.button("Ask")

def extract_text_from_image(image):
    """Extract text from an image using Gemini"""
    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    prompt = """Extract all visible text from the provided image. Return the text as a single string."""
    try:
        response = model.generate_content(
            [prompt, image],
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1000,
            )
        )
        logging.debug(f"Image text extraction response: {response.text}")
        return response.text.strip() if response.text else ""
    except Exception as e:
        st.warning(f"Error extracting text from image: {e}")
        logging.error(f"Image text extraction error: {e}")
        return ""

def describe_image(image):
    """Generate a detailed description of the image using Gemini"""
    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    prompt = """Provide a detailed description of the image, including objects, people, locations, actions, and any relevant context or relationships. Focus on elements that could help answer questions about the scene or content."""
    try:
        response = model.generate_content(
            [prompt, image],
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000,
            )
        )
        logging.debug(f"Image description response: {response.text}")
        return response.text.strip() if response.text else ""
    except Exception as e:
        st.warning(f"Error describing image: {e}")
        logging.error(f"Image description error: {e}")
        return ""

@st.cache_resource
def process_documents(_files):
    all_chunks = []
    all_images = []
    
    # Process each uploaded file
    for file in _files:
        file_extension = os.path.splitext(file.name)[1].lower()
        
        if file_extension == ".pdf":
            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = tmp_file.name
            
            # Load and split text from PDF
            try:
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
                logging.info(f"Extracted {len(chunks)} text chunks from PDF: {file.name}")
            except Exception as e:
                st.warning(f"Failed to extract text from {file.name}: {e}")
                logging.error(f"PDF text extraction error: {e}")
            
            # Extract images from PDF
            try:
                images = convert_from_path(tmp_file_path)
                for i, img in enumerate(images):
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    all_images.append({"image": img_byte_arr.getvalue(), "source": f"{file.name}_page_{i+1}"})
                    # Extract text and description from PDF images
                    image = Image.open(io.BytesIO(img_byte_arr.getvalue()))
                    extracted_text = extract_text_from_image(image)
                    image_description = describe_image(image)
                    if extracted_text:
                        doc = Document(page_content=extracted_text, metadata={"source": f"{file.name}_page_{i+1}", "type": "image_text"})
                        chunks = text_splitter.split_documents([doc])
                        all_chunks.extend(chunks)
                    if image_description:
                        doc = Document(page_content=image_description, metadata={"source": f"{file.name}_page_{i+1}", "type": "image_description"})
                        chunks = text_splitter.split_documents([doc])
                        all_chunks.extend(chunks)
                logging.info(f"Extracted {len(images)} images from PDF: {file.name}")
            except Exception as e:
                st.warning(f"Failed to extract images from {file.name}: {e}")
                logging.error(f"PDF image extraction error: {e}")
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
        elif file_extension in [".png", ".jpg", ".jpeg"]:
            # Process uploaded image
            img_byte_arr = io.BytesIO(file.read())
            all_images.append({"image": img_byte_arr.getvalue(), "source": file.name})
            logging.info(f"Added image: {file.name}")
            
            # Extract text and description from image
            image = Image.open(io.BytesIO(img_byte_arr.getvalue()))
            extracted_text = extract_text_from_image(image)
            image_description = describe_image(image)
            if extracted_text:
                doc = Document(page_content=extracted_text, metadata={"source": file.name, "type": "image_text"})
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents([doc])
                all_chunks.extend(chunks)
            if image_description:
                doc = Document(page_content=image_description, metadata={"source": file.name, "type": "image_description"})
                chunks = text_splitter.split_documents([doc])
                all_chunks.extend(chunks)
            logging.info(f"Extracted text and description from image: {file.name}")
    
    # Log the total number of chunks and images
    logging.info(f"Total text chunks: {len(all_chunks)}")
    logging.info(f"Total images: {len(all_images)}")
    
    # Handle case where no chunks are extracted
    if not all_chunks:
        st.warning("No text or image content extracted from uploaded files. Using placeholder text for vector store.")
        logging.warning("No text chunks extracted. Adding placeholder document.")
        all_chunks = [Document(page_content="Placeholder document due to no extractable text or image content.", 
                              metadata={"source": "placeholder", "type": "placeholder"})]
    
    # Create Chroma vector store for text chunks and image descriptions
    try:
        vector_store = Chroma.from_documents(
            all_chunks, 
            embedding_model, 
            persist_directory="./chroma_db_multimodal"
        )
        logging.info("Successfully created Chroma vector store")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        logging.error(f"Vector store creation error: {e}")
        return None, None
    
    return vector_store, all_images

def query_gemini(prompt, images=None):
    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    try:
        content = [prompt]
        if images:
            content.extend(images)
        response = model.generate_content(
            content,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2000,
            )
        )
        logging.debug(f"Gemini query response: {response.text}")
        return response.text
    except Exception as e:
        st.error(f"Error querying Gemini: {e}")
        logging.error(f"Query error: {e}")
        return "Failed to generate response."

if uploaded_files and ask_button and query:
    start_time = time.time()
    
    with st.spinner("Processing documents and images..."):
        result = process_documents(uploaded_files)
        if result[0] is None:
            st.error("Failed to process inputs. Please try again.")
            st.stop()
        vector_store, all_images = result
    
    with st.spinner("Retrieving relevant chunks..."):
        # Retrieve top 3 relevant chunks
        docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        logging.info(f"Retrieved {len(docs)} relevant chunks for query: {query}")
        
        # Always include images if available
        query_images = []
        if all_images:
            for img_data in all_images:
                query_images.append(Image.open(io.BytesIO(img_data["image"])))
            logging.info(f"Included {len(query_images)} images for query processing")
    
    # Create prompt for Gemini
    prompt = f"""Based on the following context from documents and images, provide a comprehensive answer to the question.

Document and Image Context (Text and Descriptions):
{context}

Question: {query}

Please provide a detailed answer by combining insights from the text context and analyzing any provided images. Describe scenes, objects, or relationships in the images if relevant to the question. If the information is insufficient, mention what additional details might be needed.

Answer:"""
    
    with st.spinner("Generating answer..."):
        answer = query_gemini(prompt, query_images if query_images else None)
        end_time = time.time()
    
    # Display results
    st.markdown("**Answer (Multimodal Vector RAG):**")
    st.write(answer)
    st.write(f"**Response Time:** {end_time - start_time:.2f} seconds")
    
    # Display some stats
    with st.expander("Processing Statistics"):
        st.write(f"**Retrieved Chunks:** {len(docs)}")
        st.write(f"**Images Processed:** {len([f for f in uploaded_files if os.path.splitext(f.name)[1].lower() in ['.png', '.jpg', '.jpeg']])}")
        st.write(f"**PDFs Processed:** {len([f for f in uploaded_files if os.path.splitext(f.name)[1].lower() == '.pdf'])}")

# Clean up temporary files on app restart
for temp_file in [f for f in os.listdir() if f.startswith("tmp") and f.endswith(".pdf")]:
    try:
        os.remove(temp_file)
    except:
        pass