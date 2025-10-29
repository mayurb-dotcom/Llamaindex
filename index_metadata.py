import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Metadata for a processed document"""
    file_path: str
    file_hash: str
    file_size: int
    processed_at: str
    num_chunks: int


class IndexMetadata(BaseModel):
    """Metadata for the entire index"""
    created_at: str
    last_updated: str
    version: str = "1.0.0"
    total_documents: int = 0
    total_chunks: int = 0
    documents: Dict[str, DocumentMetadata] = {}
    config_hash: str = ""
    
    @classmethod
    def load(cls, file_path: Path) -> Optional['IndexMetadata']:
        """Load metadata from file"""
        if not file_path.exists():
            return None
        try:
            with open(file_path, 'r') as f:
                return cls(**json.load(f))
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return None
    
    def save(self, file_path: Path):
        """Save metadata to file"""
        with open(file_path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    def needs_reindex(self, documents_dir: Path, config_hash: str) -> bool:
        """Check if reindexing is needed"""
        # Config changed
        if self.config_hash != config_hash:
            return True
        
        # Get current documents
        current_files = {str(f.relative_to(documents_dir)): f 
                        for f in documents_dir.rglob('*') if f.is_file()}
        
        # Check for new or modified documents
        for rel_path, file_path in current_files.items():
            file_hash = self._compute_file_hash(file_path)
            
            if rel_path not in self.documents:
                return True  # New document
            
            if self.documents[rel_path].file_hash != file_hash:
                return True  # Modified document
        
        # Check for deleted documents
        if len(self.documents) != len(current_files):
            return True
        
        return False
    
    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()