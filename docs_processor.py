import os
import hashlib
import json
import psutil
import gc
import re
import logging
import numpy as np
from typing import List, Optional, Generator, Dict
from datetime import datetime
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from utils.semantic_chunk import SafeSemanticChunker
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import connections, utility
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from config import Config
from logger_config import setup_logger
from index_metadata import IndexMetadata, DocumentMetadata
from monitoring import ResourceMonitor
from utils.pdf_image_extractor import PDFImageExtractor
from utils.multimodal_processor import MultiModalProcessor  

console = Console()

def find_pdf_files(documents_dir: Path) -> List[Path]:
    """Find all PDF files in documents directory"""
    pdf_files = list(documents_dir.glob("**/*.pdf"))
    return pdf_files


class DocumentProcessor:
    """Document processor with PDF image extraction"""
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'multimodal_processor') and self.multimodal_processor:
            self.multimodal_processor.cleanup()

    def __init__(self, config: Optional[Config] = None, force_reindex: bool = False):
        """Initialize the document processor"""
        self.config = config or Config()
        self.force_reindex = force_reindex
        self.logger = setup_logger('DocumentProcessor', self.config.log_dir, self.config.log_level)
        
        self.index = None
        self.documents = []
        self.nodes = []
        self.metadata: Optional[IndexMetadata] = None
        self.processed_with_streaming = False
        
        # Initialize PDF image extractor
        self.image_extractor = PDFImageExtractor(
            extract_images=getattr(self.config, 'extract_pdf_images', True),
            min_image_size=getattr(self.config, 'min_image_size', 10000)
        )
        
        # Initialize multi-modal processor if enabled
        if getattr(self.config, 'enable_multimodal', False):
            self.multimodal_processor = MultiModalProcessor(self.config)
            self.logger.info("Multi-modal processor enabled")
        else:
            self.multimodal_processor = None
            self.logger.info("Multi-modal processor disabled")
        
        self._setup_llama_index()
        self._setup_milvus()
    
    def _setup_llama_index(self):
        """Configure LlamaIndex settings with error handling"""
        try:
            # LLM Configuration
            Settings.llm = OpenAI(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                api_key=self.config.openai_api_key
            )
            
            # Embedding Configuration
            Settings.embed_model = OpenAIEmbedding(
                model=self.config.embedding_model,
                api_key=self.config.openai_api_key
            )
            
            self.logger.info(f"LlamaIndex configured with LLM: {self.config.llm_model}, Embeddings: {self.config.embedding_model}")
            
        except Exception as e:
            self.logger.error(f"Error setting up LlamaIndex: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _setup_milvus(self):
        """Setup Milvus server with retry logic"""
        try:
            console.print("[yellow]Connecting to Milvus...[/yellow]")
            
            connections.connect(
                alias="default",
                host=self.config.milvus_host,
                port=self.config.milvus_port
            )
            
            console.print("[green]✓ Connected to Milvus[/green]")
            self.logger.info("Connected to Milvus successfully")
            
        except Exception as e:
            self.logger.error(f"Error connecting to Milvus: {e}")
            console.print(f"[red]✗ Failed to connect to Milvus: {e}[/red]")
            raise
    
    def _compute_config_hash(self) -> str:
        """Compute hash of relevant config parameters"""
        config_str = f"{self.config.chunk_size}_{self.config.chunk_overlap}_{self.config.embedding_model}_{self.config.chunking_strategy}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _process_pdfs_with_image_extraction(self) -> tuple[List[Document], List[Document]]:
        """Extract text AND images from PDFs using LlamaIndex readers
        
        Returns:
            Tuple of (text_documents, image_documents)
        """
        console.print("\n[bold cyan]📄 Processing PDFs with image extraction[/bold cyan]")
        
        pdf_files = find_pdf_files(self.config.documents_dir)
        
        if not pdf_files:
            console.print("[yellow]⚠ No PDF files found[/yellow]")
            return [], []
        
        console.print(f"[green]Found {len(pdf_files)} PDF files[/green]")
        
        # Use multi-modal processor if enabled, otherwise use simple image extractor
        if self.multimodal_processor:
            console.print("[cyan]🎨 Using GPT-4 Vision for image processing[/cyan]")
            image_documents = self.multimodal_processor.batch_process_pdfs(pdf_files)
        else:
            console.print("[cyan]📷 Using simple image extraction (no GPT-4 Vision)[/cyan]")
            image_documents = self.image_extractor.batch_process_pdfs(pdf_files)
        
        # Extract text using SimpleDirectoryReader (standard LlamaIndex method)
        console.print("\n[yellow]Extracting text from PDFs...[/yellow]")
        try:
            reader = SimpleDirectoryReader(
                input_files=[str(f) for f in pdf_files],
                filename_as_id=True
            )
            text_documents = reader.load_data()
            
            # Add metadata
            for doc in text_documents:
                if 'file_name' not in doc.metadata:
                    doc.metadata['file_name'] = doc.metadata.get('filename', 'Unknown')
                doc.metadata['processing_method'] = 'standard_pdf_text'
                doc.metadata['content_type'] = 'text'
            
            console.print(f"[green]✓ Extracted text from {len(text_documents)} PDFs[/green]")
            
        except Exception as e:
            self.logger.error(f"Error extracting PDF text: {e}")
            text_documents = []
        
        return text_documents, image_documents
    
    def _load_standard_documents(self) -> List[Document]:
        """Load non-PDF documents (TXT, DOCX, MD, etc.)"""
        try:
            # Exclude PDF files from standard loading
            excluded_files = list(self.config.documents_dir.glob("**/*.pdf"))
            
            reader = SimpleDirectoryReader(
                input_dir=str(self.config.documents_dir),
                recursive=True,
                exclude=excluded_files,
                filename_as_id=True
            )
            
            documents = reader.load_data()
            
            # Add metadata
            for doc in documents:
                if 'file_name' not in doc.metadata:
                    doc.metadata['file_name'] = doc.metadata.get('filename', 'Unknown')
                doc.metadata['processing_method'] = 'standard'
                doc.metadata['content_type'] = 'text'
            
            self.logger.info(f"Loaded {len(documents)} standard documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error loading standard documents: {e}")
            return []
    
    def load_documents(self) -> List[Document]:
        """Load all documents with PDF image extraction"""
        self.logger.info(f"Loading documents from {self.config.documents_dir}...")
        
        if not self.config.documents_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.config.documents_dir}")
        
        all_documents = []
        
        # Process PDFs with image extraction
        pdf_text_docs, pdf_image_docs = self._process_pdfs_with_image_extraction()
        all_documents.extend(pdf_text_docs)
        all_documents.extend(pdf_image_docs)
        
        console.print(f"[cyan]📊 PDF Processing Summary:[/cyan]")
        console.print(f"  - Text documents: {len(pdf_text_docs)}")
        console.print(f"  - Image documents: {len(pdf_image_docs)}")
        
        # Load standard documents (non-PDFs)
        standard_documents = self._load_standard_documents()
        all_documents.extend(standard_documents)
        console.print(f"  - Other documents: {len(standard_documents)}")
        
        if not all_documents:
            raise ValueError(
                f"No documents found in {self.config.documents_dir}. "
                "Please add some documents (PDF, TXT, DOCX, etc.)"
            )
        
        self.logger.info(f"Loaded {len(all_documents)} total documents")
        console.print(f"\n[bold green]✓ Loaded {len(all_documents)} documents total[/bold green]")
        
        return all_documents
    
    def create_chunks(self, documents: List[Document]) -> List[TextNode]:
        """Create chunks from documents using configured strategy"""
        
        strategy = self.config.chunking_strategy
        console.print(f"\n[yellow]Creating chunks using {strategy} strategy...[/yellow]")
        
        # This variable will hold the correct size limit based on the strategy
        max_size_for_validation: int
        
        if strategy == "semantic":
            console.print("[cyan]🧠 Using SafeSemanticChunker (intelligent topic-based splitting)[/cyan]")
            
            # 1. Use 'max_chunk_chars' for semantic strategy
            max_size_for_validation = self.config.max_chunk_chars 
            
            chunker = SafeSemanticChunker(
                embed_model=Settings.embed_model,
                buffer_size=self.config.semantic_buffer_size,
                breakpoint_percentile_threshold=self.config.semantic_breakpoint_threshold,
                max_chunk_chars=max_size_for_validation, # <-- Pass the correct size
                min_chunk_chars=self.config.min_chunk_chars,
                batch_size=self.config.semantic_embedding_batch_size
            )
            
            nodes = chunker.get_nodes_from_documents(documents)
            
        else:  # Default to sentence
            console.print("[cyan]📝 Using SentenceSplitter (reliable fixed-size chunks)[/cyan]")
            
            # 2. Use the standard 'chunk_size' for sentence strategy
            max_size_for_validation = self.config.chunk_size 
            
            parser = SentenceSplitter(
                chunk_size=max_size_for_validation, # <-- Pass the correct size
                chunk_overlap=self.config.chunk_overlap,
                paragraph_separator="\n\n",
                secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?"
            )
            
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
        
        # === CORRECTED VALIDATION BLOCK ===
        # This block now correctly uses 'max_size_for_validation', 
        # which was set correctly based on the strategy above.
        
        oversized = [n for n in nodes if len(n.text) > max_size_for_validation]
        
        if oversized:
            console.print(f"[yellow]⚠ Found {len(oversized)} oversized chunks, splitting...[/yellow]")
            # Pass the correct max size to the splitting function
            nodes = self._force_split_oversized_chunks(nodes, max_size_for_validation) 
        
        console.print(f"[green]✓ Created {len(nodes)} chunks using {strategy} strategy[/green]")
        
        # Final validation
        final_oversized = [n for n in nodes if len(n.text) > max_size_for_validation]
        if final_oversized:
            console.print(f"[red]❌ WARNING: {len(final_oversized)} chunks still exceed limit![/red]")
        else:
            console.print(f"[green]✓ All chunks within {max_size_for_validation} character limit[/green]")
        
        return nodes
    
    def _force_split_oversized_chunks(self, nodes: List[TextNode], max_size: int) -> List[TextNode]:
        """Force split any oversized chunks"""
        validated_nodes = []
        
        for node in nodes:
            if len(node.text) <= max_size:
                validated_nodes.append(node)
            else:
                # Split by strict character count
                text = node.text
                metadata = node.metadata.copy()
                
                # Split into max_size chunks
                for i in range(0, len(text), max_size):
                    chunk_text = text[i:i + max_size]
                    
                    # Try to break at sentence boundary if possible
                    if i + max_size < len(text):  # Not the last chunk
                        search_start = int(len(chunk_text) * 0.8)
                        for sep in ['. ', '.\n', '! ', '!\n', '? ', '?\n', '\n\n', '\n']:
                            last_sep = chunk_text.rfind(sep, search_start)
                            if last_sep != -1:
                                chunk_text = chunk_text[:last_sep + len(sep)]
                                text = text[i + last_sep + len(sep):]
                                break
                    
                    if chunk_text.strip():
                        validated_nodes.append(TextNode(
                            text=chunk_text.strip(),
                            metadata=metadata
                        ))
        
        return validated_nodes
    
    def create_new_index(self, config_hash: str) -> VectorStoreIndex:
        """Create a new index from scratch"""
        console.print("\n[bold yellow]Creating new index...[/bold yellow]")
        
        # Load documents
        self.documents = self.load_documents()
        
        # Create chunks
        self.nodes = self.create_chunks(self.documents)
        
        # Clean storage directory
        if self.config.persist_dir.exists():
            console.print("[yellow]Cleaning existing storage directory...[/yellow]")
            import shutil
            shutil.rmtree(self.config.persist_dir)
        
        self.config.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Create vector store
        vector_store = MilvusVectorStore(
            uri=f"http://{self.config.milvus_host}:{self.config.milvus_port}",
            collection_name=self.config.milvus_collection,
            dim=self.config.embedding_dimension,
            overwrite=True
        )
        
        # Create fresh storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Create index
        console.print("[yellow]Building vector index...[/yellow]")
        self.index = VectorStoreIndex(
            nodes=self.nodes,
            storage_context=storage_context,
            show_progress=True
        )
        
        # Persist index
        console.print("[yellow]Persisting index...[/yellow]")
        self.index.storage_context.persist(persist_dir=str(self.config.persist_dir))
        
        # Save metadata
        self._save_metadata(config_hash)
        
        # Display summary
        self._display_processing_summary(self.get_document_statistics())
        
        console.print("[green]✓ Index created and persisted[/green]")
        return self.index
    
    def _save_metadata(self, config_hash: str):
        """Save index metadata"""
        doc_metadata = {}
        
        for doc in self.documents:
            file_name = doc.metadata.get('file_name', 'Unknown')
            
            # Compute file hash
            file_path = doc.metadata.get('file_path', '')
            file_hash = ""
            if file_path and Path(file_path).exists():
                file_hash = self._compute_file_hash(Path(file_path))
            else:
                file_hash = hashlib.md5(doc.text.encode()).hexdigest()
            
            # Count chunks for this document
            chunk_count = len([n for n in self.nodes if n.metadata.get('file_name') == file_name])
            
            doc_metadata[file_name] = DocumentMetadata(
                file_path=file_path,
                file_name=file_name,
                file_hash=file_hash,
                file_size=doc.metadata.get('file_size', len(doc.text)),
                processed_at=datetime.now().isoformat(),
                num_chunks=chunk_count,
                processing_method=doc.metadata.get('processing_method', 'standard')
            )
        
        self.metadata = IndexMetadata(
            total_documents=len(self.documents),
            total_chunks=len(self.nodes),
            config_hash=config_hash,
            documents=doc_metadata
        )
        
        self.metadata.save(self.config.metadata_file)
        self.logger.info(f"Metadata saved to {self.config.metadata_file}")
    
    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _display_processing_summary(self, stats: Dict):
        """Display processing summary"""
        table = Table(title="Document Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Documents", str(stats.get('total_documents', 0)))
        table.add_row("PDF Text Documents", str(stats.get('pdf_text_documents', 0)))
        table.add_row("PDF Image Documents", str(stats.get('pdf_image_documents', 0)))
        table.add_row("Other Documents", str(stats.get('other_documents', 0)))
        table.add_row("Total Chunks", str(len(self.nodes)))
        
        console.print(table)
    
    def get_document_statistics(self) -> Dict:
        """Get detailed statistics about processed documents"""
        if not self.documents:
            return {'total_documents': 0}
        
        pdf_text_docs = len([d for d in self.documents if d.metadata.get('processing_method') == 'standard_pdf_text'])
        pdf_image_docs = len([d for d in self.documents if d.metadata.get('processing_method') == 'image_extracted'])
        other_docs = len([d for d in self.documents if d.metadata.get('processing_method') == 'standard'])
        
        stats = {
            'total_documents': len(self.documents),
            'pdf_text_documents': pdf_text_docs,
            'pdf_image_documents': pdf_image_docs,
            'other_documents': other_docs
        }
        
        return stats
    
    def load_or_create_index(self) -> VectorStoreIndex:
        """Load existing index or create new one"""
        try:
            if (self.config.persist_dir / "docstore.json").exists():
                console.print("[yellow]Loading existing index...[/yellow]")
                
                if self.config.metadata_file.exists():
                    self.metadata = IndexMetadata.load(self.config.metadata_file)
                
                vector_store = MilvusVectorStore(
                    uri=f"http://{self.config.milvus_host}:{self.config.milvus_port}",
                    collection_name=self.config.milvus_collection,
                    dim=self.config.embedding_dimension
                )
                
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=str(self.config.persist_dir)
                )
                
                self.index = load_index_from_storage(storage_context)
                console.print("[green]✓ Index loaded successfully[/green]")
                
                return self.index
            else:
                console.print("[yellow]No existing index found, creating new one...[/yellow]")
                return self.create_new_index(self._compute_config_hash())
                
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
            console.print(f"[yellow]Failed to load index, creating new one...[/yellow]")
            return self.create_new_index(self._compute_config_hash())
    
    def get_query_engine(self, **kwargs):
        """Get query engine with custom parameters"""
        if not self.index:
            try:
                self.logger.warning("Index not set, attempting to load...")
                self.load_or_create_index()
            except Exception as e:
                raise ValueError("Index not created. Call load_or_create_index() first.")
        
        return self.index.as_query_engine(
            similarity_top_k=kwargs.get('similarity_top_k', self.config.similarity_top_k),
            **kwargs
        )
    
    def get_nodes(self) -> List[TextNode]:
        """Get nodes from the index"""
        if not self.index:
            self.logger.warning("Index not loaded, attempting to load...")
            try:
                self.load_or_create_index()
            except Exception as e:
                self.logger.error(f"Failed to load index: {e}")
                return []
    
        # If we have nodes stored in memory, return them
        if self.nodes:
            self.logger.info(f"Returning {len(self.nodes)} nodes from memory")
            return self.nodes
        
        all_nodes = []
        try:
            # Try to get nodes from docstore first
            docstore = self.index.docstore
            
            # Get all document IDs from docstore
            if hasattr(docstore, 'docs'):
                node_ids = list(docstore.docs.keys())
                self.logger.info(f"Found {len(node_ids)} node IDs in docstore")
                
                for node_id in node_ids:
                    try:
                        node = docstore.get_node(node_id)
                        if isinstance(node, TextNode):
                            all_nodes.append(node)
                    except Exception as e:
                        self.logger.debug(f"Could not retrieve node {node_id}: {e}")
        except Exception as e:
            self.logger.error(f"Error accessing docstore: {e}")
        
        # If docstore is empty, try vector retrieval with ALL chunks
        if not all_nodes:
            self.logger.info("No nodes from docstore, trying vector retriever...")
            
            # Get total chunk count from metadata first
            expected_chunks = self.metadata.total_chunks if self.metadata else 10000
            self.logger.info(f"Expected chunks from metadata: {expected_chunks}")
            
            try:
                # Use a very high similarity_top_k to get ALL chunks
                # Add a buffer to ensure we get everything
                retrieval_limit = max(expected_chunks + 100, 10000)
                
                self.logger.info(f"Retrieving up to {retrieval_limit} chunks from vector store...")
                retriever = self.index.as_retriever(similarity_top_k=retrieval_limit)
                
                # Use a generic query to retrieve all indexed content
                retrieved = retriever.retrieve("document content summary information")
                
                # Extract unique nodes
                seen = set()
                for result in retrieved:
                    if result.node_id not in seen:
                        if isinstance(result.node, TextNode):
                            all_nodes.append(result.node)
                            seen.add(result.node_id)
                
                self.logger.info(f"Retrieved {len(all_nodes)} unique nodes via retriever")
                
                # Verify we got all chunks
                if self.metadata and len(all_nodes) < self.metadata.total_chunks:
                    self.logger.warning(
                        f"Retrieved {len(all_nodes)} chunks but metadata shows {self.metadata.total_chunks}. "
                        f"Some chunks may be missing."
                    )
                
            except Exception as e:
                self.logger.error(f"Retrieval failed: {e}")
        
        # Last resort: Show metadata summary
        if not all_nodes and self.metadata:
            self.logger.warning("Attempting to reconstruct chunk info from metadata...")
            console.print(f"\n[yellow]Found metadata for {self.metadata.total_chunks} chunks[/yellow]")
            console.print(f"[yellow]Total documents: {self.metadata.total_documents}[/yellow]")
            
            # Display metadata summary
            from rich.table import Table
            table = Table(title="Indexed Documents (from metadata)")
            table.add_column("Document", style="cyan")
            table.add_column("Chunks", style="green", justify="right")
            table.add_column("Processing Method", style="yellow")
            
            for doc_name, doc_meta in self.metadata.documents.items():
                table.add_row(
                    doc_name[:50] + "..." if len(doc_name) > 50 else doc_name,
                    str(doc_meta.num_chunks),
                    doc_meta.processing_method
                )
            
            console.print(table)
            return []
        
        if all_nodes:
            self.logger.info(f"Successfully retrieved {len(all_nodes)} nodes")
        else:
            self.logger.warning("No nodes found in index")
        
        return all_nodes
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        if self.metadata:
            return {
                'total_documents': self.metadata.total_documents,
                'total_chunks': self.metadata.total_chunks
            }
        return {
            'total_documents': 0,
            'total_chunks': 0
        }
    

def main():
    """Main execution with PDF image extraction"""
    console.print("\n[bold blue]LlamaIndex Document Processor with PDF Image Extraction[/bold blue]\n")
    
    try:
        config = Config()
        logger = setup_logger('Main', config.log_dir, config.log_level)
        
        logger.info("Starting document processing with PDF image extraction...")
        
        processor = DocumentProcessor(config, force_reindex=False)
        processor.load_or_create_index()
        
        stats = processor.get_document_statistics()
        console.print(f"\n[bold green]✅ Processing Complete![/bold green]")
        console.print(f"[dim]Total documents: {stats.get('total_documents', 0)}[/dim]")
        console.print(f"[dim]PDF text: {stats.get('pdf_text_documents', 0)}[/dim]")
        console.print(f"[dim]PDF images: {stats.get('pdf_image_documents', 0)}[/dim]")
        
        logger.info("Document processing completed successfully")
        
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()