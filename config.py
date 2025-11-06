import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from typing import Optional

load_dotenv()


class AppConfig(BaseModel):
    """Robust configuration with validation"""
    
    # OpenAI Configuration
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    
    # Model Configuration
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    llm_temperature: float = 0.0001
    
    # Milvus Configuration
    milvus_host: str = Field(default="localhost", description="Milvus server host")
    milvus_port: int = Field(default=19530, description="Milvus server port")
    milvus_collection_name: str = "document_collection"
    milvus_collection: str = Field(default="document_collection", description="Milvus collection name")
    milvus_uri: str = Field(default="", description="Milvus server URI")
    milvus_token: str = Field(default="", description="Milvus authentication token (optional)")
    embedding_dimension: int = Field(default=3072, description="Embedding dimension for text-embedding-3-large")
    
    # Document Processing Configuration
    documents_dir: Path = Path("./documents")
    persist_dir: Path = Path("./storage")
    
    # CHANGED: Use sentence chunking by default (more reliable)
    chunk_size: int = 1024  # Guaranteed max size
    chunk_overlap: int = 200
    
    # Chunking Strategy Configuration
    chunking_strategy: str = Field(
        default_factory=lambda: os.getenv("CHUNKING_STRATEGY", "sentence"),
        description="Chunking strategy: 'sentence' (recommended) or 'semantic'"
    )
    
    # Semantic Chunking Configuration (for advanced users)
    semantic_buffer_size: int = Field(
        default=1, 
        description="Number of sentences to group (1=fine-grained)"
    )
    semantic_breakpoint_threshold: int = Field(
        default=95,
        description="Percentile threshold for semantic breaks (higher = larger chunks)"
    )
    
    # Semantic chunking safety limits (used by SafeSemanticChunker)
    max_chunk_chars: int = Field(
        default=2048,  # CHANGED: Match chunk_size for consistency
        description="Maximum characters per chunk for semantic chunking"
    )
    min_chunk_chars: int = Field(
        default=200,
        description="Minimum characters per chunk for semantic chunking"
    )
    semantic_embedding_batch_size: int = Field(
        default=100,
        description="Batch size for semantic chunking embeddings"
    )
    
    # Multi-Modal Configuration
    enable_multimodal: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_MULTIMODAL", "false").lower() == "true",
        description="Enable GPT-4 Vision for images, tables, and charts"
    )
    multimodal_model: str = Field(
        default_factory=lambda: os.getenv("MULTIMODAL_MODEL", "gpt-4o"),
        description="Multi-modal model to use"
    )
    multimodal_max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MULTIMODAL_MAX_TOKENS", "1024")),
        description="Max tokens for multi-modal responses"
    )
    
    # Image Classification Thresholds
    complex_image_size_threshold: int = Field(
        default=50000,
        description="Image size threshold (bytes) for classification"
    )
    
    # Retrieval Configuration
    similarity_top_k: int = Field(default=15, description="Number of top similar results to retrieve")
    
    # Citation Configuration
    citation_style: str = Field(default="minimal", description="Citation style: minimal, detailed, or none")
    max_citations: int = Field(default=3, description="Maximum number of citations to display")
    show_confidence: bool = Field(default=True, description="Show confidence indicators")
    min_answer_length: int = Field(default=150, description="Minimum characters for detailed answers")
    
    # Retry Configuration
    max_retries: int = 3
    retry_delay: int = 2
    
    # Processing Configuration
    batch_size: int = 10
    enable_cache: bool = True
    
    # Logging Configuration
    log_dir: Path = Path("./logs")
    log_level: str = "INFO"
    
    # Computed fields
    metadata_file: Path = None

    # Enhanced configuration for large documents
    processing_batch_size: int = Field(default=3, description="Number of documents to process in each batch")
    max_memory_mb: int = Field(default=2048, description="Maximum memory usage before forcing cleanup")
    enable_streaming: bool = Field(default=True, description="Enable streaming processing for large documents")
    max_file_size_mb: int = Field(default=50, description="Maximum individual file size in MB")
    large_document_threshold: int = Field(default=10, description="Number of documents that triggers streaming mode")
    large_size_threshold_mb: float = Field(default=50.0, description="Total size in MB that triggers streaming mode")
    
    # PDF Image Extraction (NEW - Simple and Free!)
    extract_pdf_images: bool = Field(
        default=True,
        description="Extract images from PDFs and include in processing"
    )
    min_image_size: int = Field(
        default=10000,
        description="Minimum image size in bytes to extract (filters out small icons)"
    )
    
    # Tesseract OCR Configuration
    tesseract_cmd: Optional[str] = Field(
        default_factory=lambda: os.getenv("TESSERACT_CMD"),
        description="Path to Tesseract executable (optional, if not in PATH)"
    )
    
    @validator('max_chunk_chars')
    def validate_max_chunk_chars(cls, v, values):
        """Validate max chunk size matches chunk_size"""
        chunk_size = values.get('chunk_size', 1024)
        if v > chunk_size:
            # Auto-adjust to match chunk_size
            return chunk_size
        return v
    
    @validator('semantic_breakpoint_threshold')
    def validate_semantic_threshold(cls, v):
        """Validate semantic threshold is in safe range"""
        if not 50 <= v <= 99:
            raise ValueError(
                "Semantic breakpoint threshold should be 50-99. "
                "Lower values = more chunks, higher values = fewer but larger chunks"
            )
        return v
    
    @validator('semantic_buffer_size')
    def validate_semantic_buffer_size(cls, v):
        """Validate buffer size"""
        if not 1 <= v <= 5:
            raise ValueError("semantic_buffer_size should be between 1 and 5")
        return v
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set metadata file path after initialization
        self.metadata_file = self.persist_dir / "index_metadata.json"
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Create multi-modal temp directory
        multimodal_temp_dir = Path("./temp_multimodal_images")
        multimodal_temp_dir.mkdir(parents=True, exist_ok=True)


# Alias for backwards compatibility
Config = AppConfig