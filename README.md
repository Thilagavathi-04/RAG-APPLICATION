# 📄 Mistral 7B RAG App (Local) 

A **Retrieval-Augmented Generation (RAG)** application that allows you to ask questions about documents using locally-run Mistral 7B model via Ollama.

## ✨ Features

- 📚 **Multiple File Formats** - Supports PDF, TXT, Markdown, and DOCX files
- 📤 **File Upload** - Upload documents directly or use a default file
- 🔍 **Smart Document Search** - Uses FAISS vector similarity search to find relevant content
- 💬 **Chat Interface** - Conversational UI with full chat history
- 🤖 **Local AI** - Runs 100% locally using Ollama (no API keys needed)
- ⚡ **Efficient Caching** - Index built once and cached for fast subsequent queries
- 📊 **Progress Tracking** - Visual progress bar during document indexing
- 🎯 **Context-Aware Answers** - Retrieves top 3 relevant chunks to generate accurate answers

## 📁 Supported File Formats

- 📄 **PDF** (.pdf) - Using PyPDF2
- 📝 **Text** (.txt) - Plain text files
- 📋 **Markdown** (.md) - Markdown documents
- 📘 **Word** (.docx) - Microsoft Word documents (requires python-docx)

## 🛠️ Technology Stack

- **Streamlit** - Web interface
- **Ollama** - Local LLM inference
- **Mistral 7B** - Both embedding generation and text generation
- **FAISS** - Fast vector similarity search
- **PyPDF2** - PDF text extraction
- **python-docx** - Word document support (optional)

## 📋 Prerequisites

### 1. Install Ollama
```bash
# Download and install from: https://ollama.ai
```

### 2. Pull Mistral 7B Model
```bash
ollama pull mistral:latest
```

### 3. Verify Installation
```bash
ollama list
# Should show: mistral:latest
```

## 🚀 Installation

### 1. Clone or navigate to the project directory
```bash
cd /path/to/rag
```

### 2. Install Python dependencies
```bash
# Basic installation (PDF, TXT, MD support)
pip install streamlit ollama PyPDF2 faiss-cpu numpy

# Full installation (includes Word document support)
pip install streamlit ollama PyPDF2 faiss-cpu numpy python-docx

# OR using uv (if you have pyproject.toml)
uv sync
```

## 📖 Usage

### Method 1: Upload a File (Recommended)

1. **Run the Application**
   ```bash
   streamlit run main.py
   ```

2. **Access the App**
   Open your browser and navigate to: `http://localhost:8501`

3. **Upload Your Document**
   - Check "Upload a file" in the sidebar
   - Click "Browse files" and select your document
   - Supported formats: PDF, TXT, MD, DOCX

4. **Ask Questions**
   - Wait for indexing to complete (shown with progress bar)
   - Type your question in the chat input
   - Get AI-generated answers based on your document
   - Continue asking follow-up questions - all history is preserved!

### Method 2: Use a Default File

1. **Add Your Document**
   Place your file in the project directory and update the filename in `main.py`:
   ```python
   document_path = "ADS.pdf"  # Change to your filename
   ```

2. **Run the Application**
   ```bash
   streamlit run main.py
   ```

3. **Access and Use**
   - Uncheck "Upload a file" in sidebar (or leave it unchecked)
   - The default file will be loaded automatically
   - Start asking questions!

## 🔧 Configuration

### Adjust Chunk Size
Modify text chunking parameters in `main.py`:
```python
def chunk_text(text, chunk_size=500, chunk_overlap=50):
```

### Change Number of Retrieved Chunks
Modify the search function:
```python
def search_similar(query, index, chunks_store, k=3):  # Change k value
```

### Switch File Format Support
The app automatically detects file format based on extension:
- `.pdf` → PDF reader
- `.txt` → Text reader
- `.md` → Markdown reader (treated as text)
- `.docx` → Word document reader (requires python-docx)

### Use Different Default File
Update the document path:
```python
document_path = "your-document.pdf"  # or .txt, .md, .docx
```

## 📊 How It Works

1. **Document Loading** - PDF text is extracted using PyPDF2
2. **Text Chunking** - Document is split into 500-character chunks with 50-character overlap
3. **Embedding Generation** - Each chunk is converted to a 4096-dimensional vector using Mistral 7B
4. **Index Building** - FAISS creates a searchable vector index (cached for reuse)
5. **Query Processing** - Your question is converted to a vector
6. **Similarity Search** - Top 3 most relevant chunks are retrieved
7. **Answer Generation** - Mistral 7B generates a response using the retrieved context

## 🎯 Performance Notes

- **First Run**: Takes time to build the index (depends on PDF size)
  - Example: ~1887 chunks takes several minutes
- **Subsequent Runs**: Instant (cached index is reused)
- **Memory Usage**: Large PDFs require more RAM for embeddings
- **GPU**: Optional, but CPU-only mode works fine for most documents

## 🗂️ Project Structure

```
rag/
├── main.py           # Main application code
├── ADS.pdf          # Your PDF document
├── README.md        # This file
├── pyproject.toml   # Python dependencies
└── uv.lock          # Lock file
```

## 🐛 Troubleshooting

### Ollama Connection Error
```bash
# Check if Ollama is running
ollama list

# If not, start Ollama service
ollama serve
```

### Cache Issues
```bash
# Clear Streamlit cache
rm -rf .streamlit/cache
```

### Memory Issues
Reduce chunk size or process fewer chunks at a time by modifying the code.

### Model Not Found
```bash
# Pull the Mistral model again
ollama pull mistral:latest
```

## 🔄 Clearing Chat History

To clear the conversation and start fresh:
- Refresh the browser page (F5)
- Or add a "Clear Chat" button in the UI

## 💡 Future Enhancements

- [ ] Support multiple PDF files
- [ ] Add document upload widget
- [ ] Export chat history
- [ ] Advanced filtering options
- [ ] Support for other document formats (DOCX, TXT, etc.)
- [ ] Adjustable model parameters (temperature, top_k, etc.)

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📧 Support

For issues or questions, please create an issue in the repository.

---

