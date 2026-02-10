from PyPDF2 import PdfReader
import faiss
import numpy as np
import streamlit as st
import ollama
import os
from pathlib import Path
import time


# =========================================================
# Load different file formats
# =========================================================

def load_document(path):
    """Load document from various file formats"""
    file_extension = Path(path).suffix.lower()
    
    if file_extension == '.pdf':
        return load_pdf(path)
    elif file_extension == '.txt':
        return load_txt(path)
    elif file_extension == '.md':
        return load_txt(path)  # Markdown is plain text
    elif file_extension == '.docx':
        return load_docx(path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def load_pdf(path):
    """Load PDF files"""
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    return text


def load_txt(path):
    """Load text and markdown files"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_docx(path):
    """Load Word documents"""
    try:
        from docx import Document
        doc = Document(path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except ImportError:
        raise ImportError("python-docx is required for .docx files. Install with: pip install python-docx")


# =========================================================
# Chunk text
# =========================================================
def chunk_text(text, chunk_size=1500, chunk_overlap=150):
    """
    Split text into chunks
    - Larger chunk_size = fewer chunks = faster indexing
    - More overlap = better context preservation
    Optimized for Intel i5-2050: 1500 chars with 150 overlap
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap

    return chunks


# =========================================================
# Get embeddings (Ollama)
# =========================================================
def get_embeddings(text):
    response = ollama.embeddings(
        model="nomic-embed-text",  # Faster and better for embeddings
        prompt=text
    )
    return response["embedding"]


# =========================================================
# Build FAISS index (cached)
# =========================================================
@st.cache_resource
def build_faiss_index(pdf_path):

    start_time = time.time()
    text = load_document(pdf_path)
    chunks = chunk_text(text)

    print(f"Total chunks to process: {len(chunks)}")
    print(f"Estimated time: {len(chunks) * 1.5 / 60:.1f} minutes (on Intel i5-2050 with nomic-embed-text)")

    dimension = 768  # nomic-embed-text embedding dimension
    index = faiss.IndexFlatL2(dimension)

    vectors = []
    chunks_store = []

    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"Processing chunk {i+1}/{len(chunks)}...")
        progress_bar.progress((i + 1) / len(chunks))
        print(f"Processing chunk {i+1}/{len(chunks)}...")  # Progress feedback
        emb = get_embeddings(chunk)
        vectors.append(emb)
        chunks_store.append(chunk)

    vectors = np.array(vectors).astype("float32")
    index.add(vectors)

    progress_bar.empty()
    status_text.empty()
    print("Index building complete!")
    return index, chunks_store


# =========================================================
# Search similar chunks
# =========================================================
def search_similar(query, index, chunks_store, k=3):
    query_vector = np.array([get_embeddings(query)]).astype("float32")
    _, indices = index.search(query_vector, k)
    return [chunks_store[i] for i in indices[0]]


# =========================================================
# Generate answer using Mistral
# =========================================================
def generate_answer(query, context):

    prompt = f"""
Use the following context to answer the question clearly and accurately.

Context:
{context}

Question:
{query}
"""

    response = ollama.chat(
        model="mistral:latest",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# =========================================================
# RAG pipeline
# =========================================================
def rag_pipeline(query, index, chunks_store):

    relevant_chunks = search_similar(query, index, chunks_store)
    context = "\n\n".join(relevant_chunks)
    answer = generate_answer(query, context)

    return answer


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="Mistral RAG", layout="centered")

st.title("📄 Mistral 7B RAG App (Local)")

# Show supported file formats
st.sidebar.title("ℹ️ Supported Formats")
st.sidebar.markdown("""
- 📄 PDF (.pdf)
- 📝 Text (.txt)
- 📋 Markdown (.md)
- 📘 Word (.docx) *requires python-docx*
""")

# Show configuration
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown(f"""
**Embedding Model:** Nomic Embed Text  
**Generation Model:** Mistral 7B  
**Chunk Size:** 1500 characters  
**Overlap:** 150 characters  
**Hardware:** Intel i5-2050 (CPU)  
**Embedding Dim:** 768

**Est. Time (500pg PDF):** ~30 minutes ⚡
""")

# Option to upload file or use existing
use_upload = st.sidebar.checkbox("Upload a file", value=False)

if use_upload:
    uploaded_file = st.file_uploader(
        "Choose a document", 
        type=['pdf', 'txt', 'md', 'docx'],
        help="Upload PDF, TXT, MD, or DOCX files"
    )
    
    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        document_path = temp_path
        st.success(f"✅ Loaded: {uploaded_file.name}")
    else:
        st.info("👆 Please upload a document to get started")
        st.stop()
else:
    # Use default file
    document_path = "ADS.pdf"  # <-- your default document
    if not os.path.exists(document_path):
        st.error(f"❌ Default file '{document_path}' not found. Please upload a file or update the path.")
        st.stop()

with st.spinner("Indexing document..."):
    index, chunks_store = build_faiss_index(document_path)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if query := st.chat_input("Ask a question from the PDF"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_pipeline(query, index, chunks_store)
            st.markdown(answer)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
