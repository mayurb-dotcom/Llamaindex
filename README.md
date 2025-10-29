# LlamaIndex Document Processing & Query System

This project provides a robust pipeline for processing, indexing, and querying large sets of documents using [LlamaIndex](https://github.com/jerryjliu/llama_index), Milvus vector database, and OpenAI embeddings/LLMs. It includes resource monitoring, health checks, and rich CLI interfaces for both processing and querying.

---

## Project Structure

```
.
├── .env                      # Environment variables (e.g., OpenAI API key)
├── config.py                 # App configuration (Pydantic-based)
├── docs_processor.py         # Main document processing, indexing, and persistence logic
├── health_check.py           # System/component health checks (OpenAI, Milvus, storage)
├── index_metadata.py         # Index and document metadata models/utilities
├── logger_config.py          # Logging setup (file + rich console)
├── main.py                   # CLI for querying the indexed documents
├── monitoring.py             # Resource monitoring (CPU/memory) during processing
├── reqs.txt                  # Python dependencies
├── view_chunks.py            # CLI for viewing indexed document chunks
├── documents/                # Directory for input documents (PDF, DOCX, TXT, etc.)
├── logs/                     # Log files directory
├── storage/                  # Persistent storage for index and metadata
│   ├── docstore.json
│   ├── graph_store.json
│   ├── image__vector_store.json
│   ├── index_metadata.json
│   └── index_store.json
```

---

## Features

- **Document Processing:** Batch and streaming processing for large document sets with memory management.
- **Vector Indexing:** Uses Milvus as a vector store for efficient semantic search.
- **OpenAI Integration:** Embeddings and LLMs for chunking and querying.
- **Resource Monitoring:** Tracks CPU and memory usage during processing.
- **Health Checks:** CLI to verify OpenAI, Milvus, documents, and storage readiness.
- **Rich CLI Output:** Uses [rich](https://github.com/Textualize/rich) for progress bars, tables, and panels.
- **Logging:** Detailed logs to file and console.
- **Metadata Tracking:** Tracks per-document chunk counts, hashes, and processing times.

---

## Setup

1. **Clone the repository and install dependencies:**
    ```sh
    pip install -r reqs.txt
    ```

2. **Configure environment variables:**
    - Copy `.env` and set your `OPENAI_API_KEY`.

3. **Prepare your documents:**
    - Place your PDF, DOCX, TXT, etc. files in the `documents/` directory.

4. **Start Milvus vector database:**
    - Ensure Milvus is running (see [Milvus docs](https://milvus.io/docs/install_standalone-docker.md)).
    - Example (if using Docker Compose):
      ```sh
      docker-compose up -d
      ```

---

## Usage

### 1. Health Check

Run a comprehensive system check:
```sh
python health_check.py
```

### 2. Process Documents

Process and index your documents:
```sh
python docs_processor.py
```
- Automatically chooses streaming mode for large document sets.

### 3. Query the Index

Run interactive or batch queries:
```sh
python main.py
```

### 4. View Chunks

Display and optionally export indexed document chunks:
```sh
python view_chunks.py
```

---

## Configuration

All configuration is managed via [`config.py`](config.py) and `.env`. Key options include:
- OpenAI API key/model
- Milvus host/port/collection
- Chunk size/overlap
- Batch size, streaming thresholds
- Logging level and directories

See [`AppConfig`](config.py) for all options.

---

## Storage

- All persistent index data and metadata are stored in the [`storage/`](storage/) directory.
- Document metadata (hash, size, chunk count) is tracked in [`index_metadata.json`](storage/index_metadata.json).

---

## Logging

- Logs are written to the [`logs/`](logs/) directory and the console with rich formatting.

---

## Troubleshooting

- Use `python health_check.py` to diagnose connectivity or configuration issues.
- Check logs in the `logs/` directory for detailed error messages.

---

## License

This project is for internal/documentation/demo use. See LICENSE if provided.

---

## Credits

- [LlamaIndex](https://github.com/jerryjliu/llama_index)
- [Milvus](https://milvus.io/)
- [OpenAI](https://openai.com/)
- [Rich](https://github.com/Textualize/rich)