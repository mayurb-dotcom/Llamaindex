import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Metadata for a processed document"""
    file_path: str
    file_name: str  # ADDED
    file_hash: str
    file_size: int
    processed_at: str
    num_chunks: int
    processing_method: str = "standard"  # ADDED with default


class IndexMetadata(BaseModel):
    """Metadata for the entire index"""
    created_at: str = ""  # ADDED default
    last_updated: str = ""  # ADDED default
    version: str = "1.0.0"
    total_documents: int = 0
    total_chunks: int = 0
    documents: Dict[str, DocumentMetadata] = {}
    config_hash: str = ""
    
    def __init__(self, **data):
        """Initialize with timestamps"""
        if 'created_at' not in data or not data['created_at']:
            data['created_at'] = datetime.now().isoformat()
        if 'last_updated' not in data or not data['last_updated']:
            data['last_updated'] = datetime.now().isoformat()
        super().__init__(**data)
    
    @classmethod
    def load(cls, file_path: Path) -> Optional['IndexMetadata']:
        """Load metadata from file"""
        if not file_path.exists():
            return None
        try:
            with open(file_path, 'r') as f:
                return cls(**json.load(f))
        except Exception as e:
            logging.error(f"Error loading metadata: {e}")
            return None
    
    def save(self, file_path: Path):
        """Save metadata to file"""
        # Update last_updated timestamp
        self.last_updated = datetime.now().isoformat()
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    def needs_reindex(self, documents_dir: Path, config_hash: str) -> bool:
        """Check if reindexing is needed"""
        # Config changed
        if self.config_hash != config_hash:
            logging.info(f"Config changed: {self.config_hash} -> {config_hash}")
            return True
        
        # Check if documents directory exists
        if not documents_dir.exists():
            logging.warning(f"Documents directory not found: {documents_dir}")
            return False
        
        # Get current documents
        supported_extensions = {'.pdf', '.txt', '.docx', '.doc', '.md', '.pptx', '.ppt'}
        current_files = {}
        for f in documents_dir.rglob('*'):
            if f.is_file() and f.suffix.lower() in supported_extensions:
                rel_path = str(f.relative_to(documents_dir))
                current_files[rel_path] = f
        
        # Check for new documents
        for rel_path in current_files:
            if rel_path not in self.documents:
                logging.info(f"New document detected: {rel_path}")
                return True
        
        # Check for modified documents
        for rel_path, file_path in current_files.items():
            if rel_path in self.documents:
                current_hash = self._compute_file_hash(file_path)
                stored_hash = self.documents[rel_path].file_hash
                if current_hash != stored_hash:
                    logging.info(f"Document modified: {rel_path}")
                    return True
        
        # Check for deleted documents
        for rel_path in self.documents:
            if rel_path not in current_files:
                logging.info(f"Document deleted: {rel_path}")
                return True
        
        logging.info("No changes detected, using existing index")
        return False
    
    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()