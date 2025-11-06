# LlamaIndex 

A production-grade document processing and querying system built with LlamaIndex, featuring advanced multi-modal capabilities (text + images), intelligent semantic chunking, comprehensive cost tracking, and real-time metrics collection. Supports PDF image extraction, OCR processing, and intelligent query execution with GPT-4o.

---

## 📁 Project Structure

```
.
├── .env                          # Environment variables (API keys, configuration)
├── .gitignore                    # Git ignore patterns
├── app.py                        # FastAPI web application
├── config.py                     # Pydantic-based configuration with validation
├── docs_processor.py             # Main document processing and indexing engine
├── health_check.py               # System health diagnostics
├── index_metadata.py             # Index and document metadata models
├── logger_config.py              # Logging setup with rich console output
├── main.py                       # Interactive query CLI
├── monitoring.py                 # Resource monitoring utilities
├── view_chunks.py                # Chunk inspection and export tool
├── README.md                     # This file
├── reqs.txt                      # Python dependencies
│
├── documents/                    # Input documents directory (place PDFs, DOCX, etc.)
│
├── logs/                         # Logging output
│   └── metrics/                  # Query metrics JSON files
│       └── metrics_YYYYMMDD_HHMMSS.json
│
├── storage/                      # Persistent index storage
│   ├── docstore.json            # Document store
│   ├── graph_store.json         # Graph relationships
│   ├── image__vector_store.json # Image embeddings
│   ├── index_metadata.json      # Index metadata and versioning
│   └── index_store.json         # Vector index
│
├── temp_multimodal_images/      # Temporary multi-modal processing
├── temp_pdf_images/              # Extracted PDF images
│
├── templates/                    # Web UI templates
│   └── index.html               # FastAPI query interface
│
└── utils/                        # Utility modules
    ├── __init__.py
    ├── cost.py                   # Cost calculation for OpenAI API
    ├── custom_prompt.py          # Custom prompt templates
    ├── langsmith_tracker.py      # LangSmith integration
    ├── metrics.py                # Metrics collection and export
    ├── multimodal_processor.py   # Multi-modal content processing
    ├── pdf_image_extractor.py    # PDF image extraction
    └── semantic_chunk.py         # Safe semantic chunking implementation
```

---

## 🚀 Setup

### Prerequisites
- Python 3.8+
- Docker (for Milvus vector database)
- OpenAI API key
- (Optional) Tesseract OCR for enhanced document detection

### 1. Create Required Directories
```sh
mkdir documents logs storage temp_multimodal_images temp_pdf_images
```

### 2. Install Dependencies
```sh
pip install -r reqs.txt
```

### 3. Configure Environment
Create a `.env` file with your configuration:
```env
# Required
OPENAI_API_KEY=your-api-key-here

# Chunking Strategy Configuration
CHUNKING_STRATEGY=semantic  # Options: 'sentence' or 'semantic'
SEMANTIC_BUFFER_SIZE=1
SEMANTIC_BREAKPOINT_THRESHOLD=75  # 50-99, higher = larger chunks

# Semantic Chunking Safety
MAX_CHUNK_CHARS=2048
MIN_CHUNK_CHARS=200
SEMANTIC_EMBEDDING_BATCH_SIZE=100

# Multi-Modal Processing (GPT-4 Vision)
ENABLE_MULTIMODAL=true
MULTIMODAL_MODEL=gpt-4o
MULTIMODAL_MAX_TOKENS=1024

# PDF Image Extraction
EXTRACT_PDF_IMAGES=true
MIN_IMAGE_SIZE=10000  # Skip small icons/logos

# Optional: LangSmith Tracking
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=llamaindex

# Optional: Tesseract OCR (if not in PATH)
TESSERACT_CMD=/usr/local/bin/tesseract
```

### 4. Start Milvus Vector Database
```sh
docker-compose up -d
```

Or use standalone Milvus: [Milvus Installation Docs](https://milvus.io/docs/install_standalone-docker.md)

### 5. Add Documents
Place your PDF, DOCX, TXT, MD, PPTX files in the `documents/` directory.

---

## 📖 Usage

### 1. Health Check
Verify all system components are working:
```sh
python health_check.py
```

**Checks:**
- ✅ OpenAI API connectivity
- ✅ Milvus server status
- ✅ Documents directory
- ✅ Index storage
- ✅ OCR availability (if configured)

### 2. Process Documents
Index your documents (first-time or after updates):
```sh
python docs_processor.py
```

**Features:**
- Automatic PDF image extraction
- Smart chunking (sentence or semantic)
- Memory-efficient batch processing
- Streaming mode for large document sets
- Progress bars and detailed statistics

**Example Output:**
```
🧠 Starting semantic chunking for 396 documents
Config: buffer_size=1, threshold=75%, max_chars=2048

Processing documents with semantic chunking... ━━━━━━━━━━━━━━━ 100% 0:00:42

✨ Semantic Chunking Results
┌─────────────────────────────────┬──────────┐
│ Metric                          │    Value │
├─────────────────────────────────┼──────────┤
│ Documents Processed             │      396 │
│ Total Chunks Created            │      680 │
│ Avg Chunks per Document         │      1.7 │
│ Max Chunk Size (Enforced)       │    2,048 │
│ Chunks Requiring Split          │ 0 (0.0%) │
│ Total Processing Time           │    42.5s │
│ Processing Speed                │     9.3/s│
└─────────────────────────────────┴──────────┘

✓ All chunks within size limits!
```

### 3. Query Documents
Run interactive queries:
```sh
python main.py
```

**Features:**
- Multi-query support (enter multiple questions)
- Real-time cost tracking
- Markdown-formatted responses with citations
- Source attribution with page numbers
- Session metrics export


### 4. View Chunks
Inspect indexed document chunks:
```sh
python view_chunks.py
```

**Features:**
- Interactive menu system
- Chunk statistics and analysis
- Sample chunk viewing
- Search and filter options
- Export to text file
- Alternative retrieval via query

### 5. Web Interface (Optional)
Start the web server:
```sh
uvicorn app:app --reload --port 3000
```

Access at: `http://localhost:3000`
