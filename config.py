import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

load_dotenv()


class AppConfig(BaseModel):
    """Robust configuration with validation"""
    
    # OpenAI Configuration
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    
    # Model Configuration
    llm_model: str = "gpt-4"
    embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.1
    
    # Milvus Configuration - Updated for server
    milvus_host: str = Field(default="localhost", description="Milvus server host")
    milvus_port: int = Field(default=19530, description="Milvus server port")
    milvus_collection_name: str = "document_collection"
    milvus_uri: str = Field(default="", description="Milvus server URI (optional)")
    milvus_token: str = Field(default="", description="Milvus authentication token (optional)")
    
    # Document Processing Configuration
    documents_dir: Path = Path("./documents")
    persist_dir: Path = Path("./storage")
    chunk_size: int = 1024
    chunk_overlap: int = 200
    
    # Retrieval Configuration
    similarity_top_k: int = 5
    
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
    
    @validator('processing_batch_size')
    def validate_batch_size(cls, v):
        if v < 1 or v > 10:
            raise ValueError("Batch size should be between 1 and 10")
        return v
    
    model_config = {
        'arbitrary_types_allowed': True,
        'validate_assignment': True
    }
    
    @validator('openai_api_key')
    def validate_api_key(cls, v):
        if not v or v == "your-api-key-here":
            raise ValueError(
                "OPENAI_API_KEY must be set in .env file or environment variables"
            )
        return v
    
    @validator('documents_dir', 'persist_dir', 'log_dir')
    def create_directories(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('milvus_uri', pre=True, always=True)
    def set_milvus_uri(cls, v, values):
        if not v and 'milvus_host' in values and 'milvus_port' in values:
            return f"http://{values['milvus_host']}:{values['milvus_port']}"
        return v
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set metadata file path after initialization
        self.metadata_file = self.persist_dir / "index_metadata.json"


# Alias for backwards compatibility
Config = AppConfig