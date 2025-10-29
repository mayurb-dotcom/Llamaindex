import os
import hashlib
import json
import psutil
import gc
from typing import List, Optional, Generator
from datetime import datetime
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import connections, utility
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config import Config
from logger_config import setup_logger
from index_metadata import IndexMetadata, DocumentMetadata
from monitoring import ResourceMonitor


console = Console()


class DocumentProcessor:
    """Robust document processor with persistence and error handling"""
    
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
        
        self._setup_llama_index()
        self._setup_milvus()
    
    def _setup_llama_index(self):
        """Configure LlamaIndex settings with error handling"""
        self.logger.info("Setting up LlamaIndex...")
        
        try:
            # Configure LLM
            Settings.llm = OpenAI(
                model=self.config.llm_model,
                api_key=self.config.openai_api_key,
                temperature=self.config.llm_temperature
            )
            
            # Configure Embeddings
            Settings.embed_model = OpenAIEmbedding(
                model=self.config.embedding_model,
                api_key=self.config.openai_api_key
            )
            
            # Configure Node Parser
            Settings.node_parser = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            
            self.logger.info("LlamaIndex configured successfully")
            console.print("[green]SUCCESS[/green] LlamaIndex configured")
            
        except Exception as e:
            self.logger.error(f"Failed to setup LlamaIndex: {e}", exc_info=True)
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _setup_milvus(self):
        """Setup Milvus server with retry logic"""
        self.logger.info("Connecting to Milvus server...")
        
        try:
            # Connect to Milvus server
            connections.connect(
                alias="default",
                host=self.config.milvus_host,
                port=self.config.milvus_port,
                timeout=30  # Increased timeout for server connections
            )
            
            # Verify connection by listing collections
            collections = utility.list_collections()
            self.logger.info(f"Connected to Milvus server successfully. Found {len(collections)} collections.")
            console.print(f"[green]SUCCESS[/green] Connected to Milvus server ({len(collections)} collections)")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus server: {e}", exc_info=True)
            raise ConnectionError(
                f"Cannot connect to Milvus server at {self.config.milvus_host}:{self.config.milvus_port}. "
                f"Ensure Milvus server is running. "
                f"You can start it with: docker-compose up -d"
            )
    
    def check_milvus_health(self) -> bool:
        """Check if Milvus server is healthy"""
        try:
            # Try to list collections
            utility.list_collections()
            return True
        except Exception as e:
            self.logger.error(f"Milvus health check failed: {e}")
            return False
    
    def _compute_config_hash(self) -> str:
        """Compute hash of relevant config parameters"""
        config_str = f"{self.config.chunk_size}_{self.config.chunk_overlap}_{self.config.embedding_model}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_document_count(self) -> int:
        """Count documents in the documents directory"""
        if not self.config.documents_dir.exists():
            return 0
        
        # Count supported document files
        supported_extensions = {'.pdf', '.txt', '.docx', '.doc', '.md', '.pptx', '.ppt'}
        files = list(self.config.documents_dir.rglob('*'))
        document_files = [f for f in files if f.is_file() and f.suffix.lower() in supported_extensions]
        
        return len(document_files)

    def get_estimated_size_mb(self) -> float:
        """Estimate total size of documents in MB"""
        if not self.config.documents_dir.exists():
            return 0.0
        
        supported_extensions = {'.pdf', '.txt', '.docx', '.doc', '.md', '.pptx', '.ppt'}
        files = list(self.config.documents_dir.rglob('*'))
        document_files = [f for f in files if f.is_file() and f.suffix.lower() in supported_extensions]
        
        total_size = sum(f.stat().st_size for f in document_files)
        return total_size / (1024 * 1024)  # Convert to MB

    def load_or_create_index(self) -> VectorStoreIndex:
        """Load existing index or create new one - FIXED VERSION"""
        self.logger.info("Checking for existing index...")
        
        # Load metadata first
        self.metadata = IndexMetadata.load(self.config.metadata_file)
        config_hash = self._compute_config_hash()
        
        # Check if we should load existing index
        should_load_existing = (
            not self.force_reindex and 
            self.metadata is not None and
            not self.metadata.needs_reindex(self.config.documents_dir, config_hash)
        )
        
        if should_load_existing:
            try:
                self.logger.info("Loading existing index from storage...")
                console.print("[yellow]Loading existing index...[/yellow]")
                
                # Create vector store for Milvus server
                vector_store = MilvusVectorStore(
                    uri=self.config.milvus_uri,
                    token=self.config.milvus_token if self.config.milvus_token else None,
                    collection_name=self.config.milvus_collection_name,
                    dim=1536,
                    overwrite=False
                )
                
                # Check if storage files exist
                storage_files_exist = (
                    (self.config.persist_dir / "docstore.json").exists() and
                    (self.config.persist_dir / "index_store.json").exists() and
                    (self.config.persist_dir / "graph_store.json").exists()
                )
                
                if not storage_files_exist:
                    self.logger.warning("Storage files not found, will create new index")
                    raise FileNotFoundError("Storage files not found")
                
                # Verify Milvus collection exists
                if not utility.has_collection(self.config.milvus_collection_name):
                    self.logger.warning("Milvus collection not found, will create new index")
                    raise ValueError("Milvus collection not found")
                
                # Create storage context
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=str(self.config.persist_dir)
                )
                
                # Load index
                self.index = load_index_from_storage(storage_context)
                
                # Verify index loaded properly by checking nodes
                if hasattr(self.index, '_vector_store') and self.index._vector_store:
                    self.logger.info("Index loaded successfully with vector store")
                else:
                    self.logger.warning("Index loaded but vector store issue detected")
                    raise ValueError("Index loading issue")
                
                self.logger.info(f"Loaded existing index with {self.metadata.total_chunks} chunks")
                console.print(f"[green]SUCCESS[/green] Loaded existing index with {self.metadata.total_chunks} chunks")
                
                return self.index
                
            except Exception as e:
                self.logger.warning(f"Failed to load existing index: {e}. Creating new index...")
                console.print(f"[yellow]Failed to load existing index: {e}. Creating new one...[/yellow]")
                # Continue to create new index
        
        # Create new index if loading failed or conditions not met
        return self.create_new_index(config_hash)
        
    def clear_storage(self):
        """Clear storage directory to force fresh index creation"""
        import shutil
        
        if self.config.persist_dir.exists():
            self.logger.info(f"Clearing storage directory: {self.config.persist_dir}")
            shutil.rmtree(self.config.persist_dir)
        
        # Recreate the directory
        self.config.persist_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Storage directory cleared")
    
    def create_new_index(self, config_hash: str) -> VectorStoreIndex:
        """Create a new index from scratch"""
        self.logger.info("Creating new index...")
        console.print("[yellow]Creating new index...[/yellow]")
        
        # Clear any existing storage files to avoid conflicts
        self.clear_storage()
        
        # Drop existing collection if force reindex
        if self.force_reindex and utility.has_collection(self.config.milvus_collection_name):
            utility.drop_collection(self.config.milvus_collection_name)
            self.logger.info(f"Dropped existing collection: {self.config.milvus_collection_name}")
        
        # Load documents
        self.documents = self.load_documents()
        
        # Process and create index
        self.index = self.process_and_index()
        
        # Save metadata
        self._save_metadata(config_hash)
        
        return self.index
    
    def load_documents(self) -> List:
        """Load documents with progress tracking"""
        self.logger.info(f"Loading documents from {self.config.documents_dir}...")
        
        if not self.config.documents_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.config.documents_dir}")
        
        files = list(self.config.documents_dir.rglob('*'))
        files = [f for f in files if f.is_file()]
        
        if not files:
            raise ValueError(
                f"No documents found in {self.config.documents_dir}. "
                "Please add some documents (PDF, TXT, DOCX, etc.)"
            )
        
        self.logger.info(f"Found {len(files)} file(s)")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Loading documents...", total=len(files))
            
            reader = SimpleDirectoryReader(
                input_dir=str(self.config.documents_dir),
                recursive=True
            )
            documents = reader.load_data()
            
            progress.update(task, completed=len(files))
        
        self.logger.info(f"Loaded {len(documents)} document(s)")
        console.print(f"[green]SUCCESS[/green] Loaded {len(documents)} document(s)")
        
        for doc in documents:
            filename = doc.metadata.get('file_name', 'Unknown')
            self.logger.debug(f"  - {filename} ({len(doc.text)} chars)")
        
        return documents
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    @retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def process_and_index(self) -> VectorStoreIndex:
        """Process documents with retry logic"""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        self.logger.info("Processing documents and creating index...")
        console.print("[yellow]Processing documents and creating embeddings...[/yellow]")
        
        try:
            # Create vector store for Milvus server
            vector_store = MilvusVectorStore(
                uri=self.config.milvus_uri,
                token=self.config.milvus_token if self.config.milvus_token else None,
                collection_name=self.config.milvus_collection_name,
                dim=1536,
                overwrite=True
            )
            
            # Create EMPTY storage context for new index
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            
            # Create index
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Creating embeddings and indexing...", total=None)
                
                index = VectorStoreIndex.from_documents(
                    self.documents,
                    storage_context=storage_context,
                    show_progress=False
                )
                
                progress.update(task, completed=100)
            
            # Store nodes
            self.nodes = Settings.node_parser.get_nodes_from_documents(self.documents)
            
            # Persist index - this creates the storage files
            storage_context.persist(persist_dir=str(self.config.persist_dir))
            
            # Save metadata BEFORE any cleanup
            config_hash = self._compute_config_hash()
            self._save_metadata(config_hash)
            
            self.logger.info(f"Created and persisted index with {len(self.nodes)} chunks from {len(self.documents)} documents")
            console.print(f"[green]SUCCESS[/green] Created index with {len(self.nodes)} chunks from {len(self.documents)} documents")
            
            return index
            
        except Exception as e:
            self.logger.error(f"Failed to process documents: {e}", exc_info=True)
            raise
    
    def _save_metadata(self, config_hash: str):
        """Save index metadata with proper document tracking"""
        doc_metadata = {}
        
        # Ensure persist directory exists
        self.config.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Method 1: Use self.documents if available
        if self.documents:
            for doc in self.documents:
                file_name = doc.metadata.get('file_name', 'unknown')
                file_path = self.config.documents_dir / file_name
                
                if file_path.exists():
                    doc_metadata[file_name] = DocumentMetadata(
                        file_path=str(file_path),
                        file_hash=IndexMetadata._compute_file_hash(file_path),
                        file_size=file_path.stat().st_size,
                        processed_at=datetime.now().isoformat(),
                        num_chunks=len([n for n in self.nodes if n.metadata.get('file_name') == file_name])
                    )
        # Method 2: Scan documents directory if self.documents is empty
        else:
            supported_extensions = {'.pdf', '.txt', '.docx', '.doc', '.md', '.pptx', '.ppt'}
            files = list(self.config.documents_dir.rglob('*'))
            document_files = [f for f in files if f.is_file() and f.suffix.lower() in supported_extensions]
            
            for file_path in document_files:
                file_name = file_path.name
                doc_metadata[file_name] = DocumentMetadata(
                    file_path=str(file_path),
                    file_hash=IndexMetadata._compute_file_hash(file_path),
                    file_size=file_path.stat().st_size,
                    processed_at=datetime.now().isoformat(),
                    num_chunks=0  # We can't determine this without processing
                )
        
        # Calculate total_documents from the documents we found
        total_documents = len(doc_metadata)
        
        # Calculate total_chunks from nodes if available, otherwise use metadata
        if self.nodes:
            total_chunks = len(self.nodes)
        elif self.metadata:
            total_chunks = self.metadata.total_chunks
        else:
            total_chunks = 0
        
        now = datetime.now().isoformat()
        self.metadata = IndexMetadata(
            created_at=now,
            last_updated=now,
            total_documents=total_documents,
            total_chunks=total_chunks,
            documents=doc_metadata,
            config_hash=config_hash
        )
        
        # Ensure metadata file directory exists
        self.config.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        self.metadata.save(self.config.metadata_file)
        self.logger.info(f"Saved metadata to {self.config.metadata_file} with {total_documents} documents and {total_chunks} chunks")
        
    def get_query_engine(self, **kwargs):
        """Get query engine with custom parameters"""
        if not self.index:
            # Try to load the index if it's not set
            try:
                self.logger.warning("Index not set, attempting to load...")
                self.load_or_create_index()
            except:
                raise ValueError("Index not created. Call load_or_create_index() or process_large_document_set() first.")
        
        return self.index.as_query_engine(
            similarity_top_k=kwargs.get('similarity_top_k', self.config.similarity_top_k),
            **kwargs
        )
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        if not self.metadata:
            return {}
        
        return {
            'total_documents': self.metadata.total_documents,
            'total_chunks': self.metadata.total_chunks,
            'created_at': self.metadata.created_at,
            'last_updated': self.metadata.last_updated,
            'version': self.metadata.version,
            'documents': list(self.metadata.documents.keys())
        }
    
    def get_nodes(self):
        """Get nodes, loading if necessary"""
        if not self.nodes and self.index:
            try:
                # For vector stores that store text directly (like Milvus),
                # we need to retrieve nodes differently
                if hasattr(self.index, 'vector_store'):
                    # Try to get nodes from the vector store directly
                    vector_store = self.index.vector_store
                    
                    # Method 1: Try to use the vector store's client to query all data with limit
                    if hasattr(vector_store, '_milvusclient'):
                        try:
                            milvus_client = vector_store._milvusclient
                            
                            # Query all entities from the collection WITH LIMIT
                            results = milvus_client.query(
                                collection_name=self.config.milvus_collection_name,
                                output_fields=["id", "text", "metadata", "doc_id"],
                                limit=1000  # Add limit to avoid the empty expression error
                            )
                            
                            self.nodes = []
                            for result in results:
                                # Parse metadata if it's stored as string
                                metadata = result.get("metadata", {})
                                if isinstance(metadata, str):
                                    try:
                                        metadata = json.loads(metadata)
                                    except:
                                        metadata = {"raw_metadata": metadata}
                                
                                node = TextNode(
                                    text=result.get("text", ""),
                                    id_=result.get("id", ""),
                                    metadata=metadata
                                )
                                self.nodes.append(node)
                            
                            self.logger.info(f"Retrieved {len(self.nodes)} nodes from Milvus")
                            
                        except Exception as e:
                            self.logger.warning(f"Could not retrieve nodes from Milvus directly: {e}")
                    
                    # Method 2: If direct query fails, try to get nodes via document parser
                    if not self.nodes and self.documents:
                        self.logger.info("Reconstructing nodes from documents")
                        self.nodes = Settings.node_parser.get_nodes_from_documents(self.documents)
                
                # Method 3: Final fallback - create minimal nodes from metadata
                if not self.nodes and self.metadata:
                    self.logger.info("Creating placeholder nodes from metadata")
                    self._create_placeholder_nodes()
                        
            except Exception as e:
                self.logger.error(f"Error retrieving nodes: {e}")
                # Return empty list as fallback
                self.nodes = []
        
        return self.nodes

    def _create_placeholder_nodes(self):
        """Create placeholder nodes when real nodes can't be retrieved"""
        self.nodes = []
        total_chunks = self.metadata.total_chunks if self.metadata else 22
        
        for i in range(total_chunks):
            node = TextNode(
                text=f"Chunk {i+1} - Content stored in Milvus vector database. " +
                    f"Actual text content is retrieved during query operations. " +
                    f"This is chunk {i+1} of {total_chunks} total chunks.",
                id_=f"chunk_{i+1}",
                metadata={
                    "source": "milvus_vector_store", 
                    "chunk_id": i+1,
                    "total_chunks": total_chunks,
                    "note": "Content retrieved directly from vector store during queries"
                }
            )
            self.nodes.append(node)

    def load_documents_streaming(self) -> Generator:
        """Load documents in batches to manage memory"""
        self.logger.info(f"Loading documents from {self.config.documents_dir}...")
        
        if not self.config.documents_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.config.documents_dir}")
        
        files = list(self.config.documents_dir.rglob('*'))
        files = [f for f in files if f.is_file() and f.suffix.lower() in ['.pdf', '.txt', '.docx', '.doc']]
        
        if not files:
            raise ValueError(f"No supported documents found in {self.config.documents_dir}")
        
        self.logger.info(f"Found {len(files)} document(s)")
        
        # Process in batches
        batch_size = min(self.config.processing_batch_size, len(files))
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(files)-1)//batch_size + 1}")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"Loading batch {i//batch_size + 1}...", total=len(batch_files))
                
                reader = SimpleDirectoryReader(
                    input_files=[str(f) for f in batch_files]
                )
                batch_documents = reader.load_data()
                
                progress.update(task, completed=len(batch_files))
                
                yield batch_documents
                
                # Clean up memory
                del batch_documents
                gc.collect()
    
    @retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def process_and_index_streaming(self) -> VectorStoreIndex:
        """Process documents in batches with memory management"""
        if not self.config.documents_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.config.documents_dir}")
        
        self.logger.info("Processing documents in streaming mode...")
        console.print("[yellow]Processing documents in batches...[/yellow]")
        
        try:
            # Create vector store for Milvus server
            vector_store = MilvusVectorStore(
                uri=self.config.milvus_uri,
                token=self.config.milvus_token if self.config.milvus_token else None,
                collection_name=self.config.milvus_collection_name,
                dim=1536,
                overwrite=True
            )
            
            # Create EMPTY storage context for new index
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            
            all_nodes = []
            all_documents = []  # Track all documents processed
            total_documents = 0
            index = None
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console
            ) as progress:
                main_task = progress.add_task("Processing all documents...", total=None)
                
                # Process documents in batches
                for batch_num, batch_documents in enumerate(self.load_documents_streaming(), 1):
                    # Monitor memory usage
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                    self.logger.info(f"Batch {batch_num} - Memory usage: {memory_usage:.2f} MB")
                    
                    if memory_usage > self.config.max_memory_mb:
                        self.logger.warning("High memory usage, forcing garbage collection")
                        gc.collect()
                    
                    progress.update(main_task, description=f"Processing batch {batch_num}")
                    
                    # Store documents for metadata
                    all_documents.extend(batch_documents)
                    
                    # Create index for this batch or add to existing
                    if batch_num == 1:
                        index = VectorStoreIndex.from_documents(
                            batch_documents,
                            storage_context=storage_context,
                            show_progress=False
                        )
                    else:
                        # Add documents to existing index
                        for doc in batch_documents:
                            index.insert(doc)
                    
                    # Store nodes for this batch
                    batch_nodes = Settings.node_parser.get_nodes_from_documents(batch_documents)
                    all_nodes.extend(batch_nodes)
                    total_documents += len(batch_documents)
                    
                    # Update progress
                    progress.update(main_task, advance=len(batch_documents))
                    
                    # Clean up - but keep document metadata
                    del batch_documents
                    del batch_nodes
                    gc.collect()
                
                progress.update(main_task, description="Finalizing index...")
            
            # Store all nodes and documents
            self.nodes = all_nodes
            self.documents = all_documents  # Keep documents for metadata saving
            
            # Persist index
            if index:
                storage_context.persist(persist_dir=str(self.config.persist_dir))
            
            # SAVE METADATA - This is critical!
            config_hash = self._compute_config_hash()
            self._save_metadata(config_hash)
            
            self.logger.info(f"Created and persisted index with {len(self.nodes)} chunks from {total_documents} documents")
            console.print(f"[green]SUCCESS[/green] Created index with {len(self.nodes)} chunks from {total_documents} documents")
            
            return index
            
        except Exception as e:
            self.logger.error(f"Failed to process documents: {e}", exc_info=True)
            raise

    def process_large_document_set(self) -> VectorStoreIndex:
        """Process large document sets with enhanced error handling"""
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        try:
            console.print("\n[bold blue]Starting large document processing[/bold blue]")
            
            # Check document count and size
            doc_count = self.get_document_count()
            total_size_mb = self.get_estimated_size_mb()
            console.print(f"[dim]Processing {doc_count} documents ({total_size_mb:.1f} MB)[/dim]")
            
            # Check available disk space
            storage_info = psutil.disk_usage(str(self.config.persist_dir))
            if storage_info.free < 500 * 1024 * 1024:  # 500MB minimum
                console.print("[yellow]Warning: Low disk space available[/yellow]")
            
            # Process documents using streaming and store the result in self.index
            self.index = self.process_and_index_streaming()
            
            # Mark that we've processed with streaming
            self.processed_with_streaming = True
            
            # Show resource report
            console.print("\n[bold green]Processing completed![/bold green]")
            console.print(monitor.get_report())
            
            return self.index
            
        except MemoryError:
            console.print("[red]FAIL Memory error: Consider reducing batch size or processing fewer documents at once[/red]")
            raise
        except Exception as e:
            console.print(f"[red]FAIL Processing error: {e}[/red]")
            raise
        finally:
            monitor.stop_monitoring()

    def load_existing_index_only(self) -> VectorStoreIndex:
        """Load existing index without any processing - read-only mode"""
        self.logger.info("Loading existing index in read-only mode...")
        
        # Load metadata
        self.metadata = IndexMetadata.load(self.config.metadata_file)
        
        if not self.metadata:
            raise ValueError("No existing index metadata found. Please process documents first.")
        
        try:
            # Create vector store for Milvus server
            vector_store = MilvusVectorStore(
                uri=self.config.milvus_uri,
                token=self.config.milvus_token if self.config.milvus_token else None,
                collection_name=self.config.milvus_collection_name,
                dim=1536,
                overwrite=False
            )
            
            # Check if storage files exist
            storage_files_exist = (
                (self.config.persist_dir / "docstore.json").exists() and
                (self.config.persist_dir / "index_store.json").exists()
            )
            
            if not storage_files_exist:
                raise FileNotFoundError("Storage files not found. Please process documents first.")
            
            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(self.config.persist_dir)
            )
            
            # Load index
            self.index = load_index_from_storage(storage_context)
            
            self.logger.info("Loaded existing index in read-only mode")
            console.print(f"[green]SUCCESS[/green] Loaded existing index with {self.metadata.total_chunks} chunks")
            
            return self.index
            
        except Exception as e:
            self.logger.error(f"Failed to load existing index: {e}")
            raise ValueError(f"Cannot load existing index: {e}. Please process documents first.")


def main():
    """Main execution with robust error handling and large document support"""
    console.print("\n[bold blue]LlamaIndex Document Processor[/bold blue]\n")
    
    try:
        # Initialize config
        config = Config()
        logger = setup_logger('Main', config.log_dir, config.log_level)
        
        logger.info("Starting document processing...")
        
        # Initialize processor
        processor = DocumentProcessor(config, force_reindex=False)
        
        # Check document count and size
        doc_count = processor.get_document_count()
        total_size_mb = processor.get_estimated_size_mb()
        
        console.print(f"[dim]Found {doc_count} documents ({total_size_mb:.1f} MB)[/dim]")
        
        # Check if metadata and index exist
        metadata_exists = config.metadata_file.exists()
        storage_exists = (
            (config.persist_dir / "docstore.json").exists() and
            (config.persist_dir / "index_store.json").exists()
        )
        
        console.print(f"[dim]Metadata exists: {metadata_exists}, Storage exists: {storage_exists}[/dim]")
        
        if doc_count == 0:
            console.print("[red]FAIL No documents found to process[/red]")
            return
        
        # Check Milvus health
        if not processor.check_milvus_health():
            console.print("[red]FAIL[/red] Milvus server is not healthy. Please check the connection.")
            return
        
        # Use streaming for large document sets
        use_streaming = (
            config.enable_streaming and 
            (doc_count > config.large_document_threshold or total_size_mb > config.large_size_threshold_mb)
        )
        
        if use_streaming:
            console.print("[yellow]Using streaming mode for large document set...[/yellow]")
            processor.process_large_document_set()
        else:
            console.print("[yellow]Using standard processing mode...[/yellow]")
            processor.load_or_create_index()
        
        # Verify index was created
        if not processor.index:
            raise ValueError("Index was not created successfully")
        
        # Display stats
        stats = processor.get_stats()
        console.print("\n[bold green]Index Statistics:[/bold green]")
        console.print(f"  Documents: {stats.get('total_documents', 0)}")
        console.print(f"  Chunks: {stats.get('total_chunks', 0)}")
        console.print(f"  Last Updated: {stats.get('last_updated', 'N/A')}")
        
        console.print("\n[bold green]SUCCESS Processing completed successfully![/bold green]\n")
        console.print("Next steps:")
        console.print("  1. Run 'python main.py' to execute queries")
        console.print("  2. Run 'python view_chunks.py' to view processed chunks")
        
        logger.info("Document processing completed successfully")
        
    except ValueError as e:
        console.print(f"[red]FAIL Configuration Error:[/red] {e}")
        logger.error(f"Configuration error: {e}")
    except ConnectionError as e:
        console.print(f"[red]FAIL Connection Error:[/red] {e}")
        logger.error(f"Connection error: {e}")
    except Exception as e:
        console.print(f"[red]FAIL Error:[/red] {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    main()