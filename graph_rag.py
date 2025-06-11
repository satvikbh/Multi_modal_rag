import streamlit as st
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import networkx as nx
from dotenv import load_dotenv
import os
import time
import json
import logging
import re
import tempfile

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
st.title("Multi-PDF Graph RAG Application")
uploaded_files = st.file_uploader("Upload up to 4 PDF documents", type="pdf", accept_multiple_files=True)
if uploaded_files and len(uploaded_files) > 4:
    st.error("Please upload no more than 4 PDF files.")
    uploaded_files = uploaded_files[:4]  # Limit to 4 files
query = st.text_input("Enter your question")
ask_button = st.button("Ask")

def clean_json_response(response_text):
    """Clean and extract JSON from Gemini response"""
    if not response_text:
        return None
    
    # Remove markdown code blocks if present
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*$', '', response_text)
    
    # Try to find JSON content between curly braces
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return None

def fallback_entity_extraction(text):
    """Fallback method to extract entities using simple patterns"""
    entities = []
    relationships = []
    
    # Simple pattern matching for common entities
    patterns = {
        'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
        'ORGANIZATION': r'\b[A-Z][A-Z\s]+\b',
        'LOCATION': r'\b(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    }
    
    for entity_type, pattern in patterns.items():
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match else ""
            if match and len(match.strip()) > 2:
                entities.append({"name": match.strip(), "type": entity_type})
    
    # Simple relationship extraction
    sentences = text.split('.')
    for sentence in sentences[:3]:  # Limit to first 3 sentences
        words = sentence.split()
        if len(words) > 5:  # Only process longer sentences
            for i, word in enumerate(words):
                if word.lower() in ['is', 'was', 'has', 'had', 'owns', 'works']:
                    if i > 0 and i < len(words) - 1:
                        source = words[max(0, i-2):i]
                        target = words[i+1:min(len(words), i+3)]
                        if source and target:
                            relationships.append({
                                "source": " ".join(source).strip(),
                                "target": " ".join(target).strip(),
                                "relation": word
                            })
    
    return {"entities": entities, "relationships": relationships}

# Function to extract entities and relationships using Gemini
def extract_entities_relations(text):
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
    # Improved prompt with more specific instructions
    prompt = f"""You are an expert at extracting entities and relationships from text. 
Extract entities and relationships from the following text and return ONLY a valid JSON object.

IMPORTANT: 
- Return ONLY the JSON object, no other text
- Do not include markdown formatting
- Ensure the JSON is properly formatted

Text to analyze:
{text[:1500]}

Required JSON format:
{{
    "entities": [
        {{"name": "entity_name", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT"}}
    ],
    "relationships": [
        {{"source": "entity1", "target": "entity2", "relation": "relationship_type"}}
    ]
}}

JSON Response:"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1000,
            )
        )
        
        # Log the raw response for debugging
        logging.debug(f"Gemini raw response: {response.text}")
        
        if not response.text:
            st.warning("Empty response from Gemini API, using fallback method.")
            logging.warning("Empty response from Gemini API.")
            return fallback_entity_extraction(text)
        
        # Clean and parse the response
        cleaned_json = clean_json_response(response.text)
        if cleaned_json:
            # Validate the structure
            if "entities" in cleaned_json and "relationships" in cleaned_json:
                return cleaned_json
            else:
                st.warning("Invalid JSON structure from Gemini, using fallback method.")
                return fallback_entity_extraction(text)
        else:
            st.warning("Could not parse Gemini response, using fallback method.")
            logging.warning(f"Could not parse response: {response.text}")
            return fallback_entity_extraction(text)
            
    except json.JSONDecodeError as e:
        st.warning(f"JSON parse error: {e}. Using fallback method.")
        logging.error(f"JSON parse error: {e}, Response: {response.text if 'response' in locals() else 'No response'}")
        return fallback_entity_extraction(text)
    except Exception as e:
        st.warning(f"Error in entity extraction: {e}. Using fallback method.")
        logging.error(f"Entity extraction error: {e}")
        return fallback_entity_extraction(text)

# Function to process multiple PDFs and create vector store and knowledge graph
@st.cache_resource
def process_document_graph(_files):
    all_chunks = []
    G = nx.DiGraph()
    all_entities = []
    all_relations = []
    
    # Process each uploaded file
    for file in _files:
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
    
    # Process only first 8 chunks to stay within rate limits
    chunks_to_process = all_chunks[:8]  # Limit chunks for free tier
    
    # Process chunks with progress tracking
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks_to_process):
        progress_bar.progress((i + 1) / len(chunks_to_process))
        
        result = extract_entities_relations(chunk.page_content)
        
        # Add entities to graph
        for entity in result.get("entities", []):
            entity_name = entity.get("name", "").strip()
            entity_type = entity.get("type", "unknown")
            if entity_name and len(entity_name) > 1:
                G.add_node(entity_name, type=entity_type)
                all_entities.append({"name": entity_name, "content": entity_name})
        
        # Add relationships to graph
        for rel in result.get("relationships", []):
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            relation = rel.get("relation", "related_to")
            
            if source and target and len(source) > 1 and len(target) > 1:
                G.add_edge(source, target, relation=relation)
                all_relations.append(f"{source} {relation} {target}")
        
        # Add delay to respect rate limits (free tier: 10 requests/minute)
        if i > 0:
            time.sleep(7)
    
    progress_bar.empty()
    
    # Create Chroma vector store for chunks and entities
    texts = [chunk.page_content for chunk in all_chunks] + [entity["content"] for entity in all_entities]
    metadata = [{"type": "chunk"} for _ in all_chunks] + [{"type": "entity"} for _ in all_entities]
    
    try:
        vector_store = Chroma.from_texts(
            texts, 
            embedding_model, 
            metadatas=metadata, 
            persist_directory="./chroma_db_graph"
        )
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None, None, None
    
    return vector_store, G, all_relations

# Function to query Gemini
def query_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
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

if uploaded_files and ask_button and query:
    start_time = time.time()
    
    with st.spinner("Processing documents and building knowledge graph..."):
        result = process_document_graph(uploaded_files)
        if result[0] is None:
            st.error("Failed to process documents. Please try again.")
            st.stop()
        vector_store, graph, relations = result
    
    with st.spinner("Retrieving relevant chunks and entities..."):
        # Retrieve top 5 relevant items (chunks or entities)
        docs = vector_store.similarity_search(query, k=5)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Get related entities from graph
        related_entities = []
        query_words = query.lower().split()
        
        for node in graph.nodes:
            if any(word in node.lower() for word in query_words) or \
               any(node.lower() in doc.page_content.lower() for doc in docs):
                neighbors = list(graph.neighbors(node))
                related_entities.extend([f"{node} -> {neighbor}" for neighbor in neighbors[:3]])
        
        # Limit graph context for brevity
        graph_context = "\n".join(related_entities[:10] + relations[:5])
    
    # Create enhanced prompt for Gemini
    prompt = f"""Based on the following context and knowledge graph information from multiple documents, provide a comprehensive answer to the question.

Document Context:
{context}

Knowledge Graph Relationships:
{graph_context}

Question: {query}

Please provide a detailed answer based on the context and relationships above. If the information is not sufficient, please mention what additional information might be needed.

Answer:"""
    
    with st.spinner("Generating answer..."):
        answer = query_gemini(prompt)
        end_time = time.time()
    
    # Display results
    st.markdown("**Answer (Multi-PDF Graph RAG):**")
    st.write(answer)
    st.write(f"**Response Time:** {end_time - start_time:.2f} seconds")
    
    # Display some stats
    with st.expander("Graph Statistics"):
        st.write(f"**Nodes (Entities):** {graph.number_of_nodes()}")
        st.write(f"**Edges (Relationships):** {graph.number_of_edges()}")
        st.write(f"**Retrieved Chunks:** {len(docs)}")
        st.write(f"**Related Entities:** {len(related_entities)}")

# Clean up temporary files on app restart
for temp_file in [f for f in os.listdir() if f.startswith("tmp") and f.endswith(".pdf")]:
    try:
        os.remove(temp_file)
    except:
        pass