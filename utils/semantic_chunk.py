"""Safe semantic chunking with size limits and batch processing"""
from typing import List, Dict
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import time

console = Console()


class SafeSemanticChunker:
    """Safe semantic chunking with automatic fallback and size validation"""
    
    def __init__(
        self,
        embed_model: OpenAIEmbedding,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 75,
        max_chunk_chars: int = 2048,
        min_chunk_chars: int = 200,
        batch_size: int = 100
    ):
        """Initialize safe semantic chunker
        
        Args:
            embed_model: Embedding model for semantic analysis
            buffer_size: Sentences to group (1-3 recommended)
            breakpoint_percentile_threshold: Semantic break threshold (50-90 recommended)
            max_chunk_chars: Maximum characters per chunk (HARD LIMIT)
            min_chunk_chars: Minimum characters per chunk
            batch_size: Batch size for processing
        """
        self.embed_model = embed_model
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.batch_size = batch_size
        
        # Create semantic splitter with CHUNK_SIZE enforcement
        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=embed_model,
            chunk_size=max_chunk_chars,  # Hard limit enforcement
            include_metadata=True
        )
        
        # Fallback sentence splitter (only for error recovery)
        self.sentence_splitter = SentenceSplitter(
            chunk_size=max_chunk_chars,
            chunk_overlap=200,
            include_metadata=True
        )
        
        self.stats = {
            "total_chunks": 0,
            "oversized_chunks_split": 0,
            "fallback_splits": 0,
            "total_processing_time": 0,
            "documents_processed": 0
        }
    
    def get_nodes_from_documents(self, documents: List[Document], show_progress: bool = True) -> List[TextNode]:
        """Chunk documents with safety checks and fallbacks
        
        Args:
            documents: Documents to chunk
            show_progress: Show progress bar
            
        Returns:
            List of safely sized chunks
        """
        console.print(f"\n[cyan]🧠 Starting semantic chunking for {len(documents)} documents[/cyan]")
        console.print(f"[dim]Config: buffer_size={self.buffer_size}, threshold={self.breakpoint_percentile_threshold}%, max_chars={self.max_chunk_chars}[/dim]")
        
        start_time = time.time()
        all_nodes = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                "Processing documents with semantic chunking...", 
                total=len(documents)
            )
            
            # Process documents in batches to manage memory and rate limits
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                total_batches = (len(documents) + self.batch_size - 1) // self.batch_size
                
                progress.update(
                    task, 
                    description=f"Processing batch {batch_num}/{total_batches} ({len(batch)} docs)..."
                )
                
                # Semantic chunking with built-in size enforcement
                try:
                    batch_nodes = self.semantic_splitter.get_nodes_from_documents(batch)
                except Exception as e:
                    console.print(f"[yellow]⚠ Batch {batch_num}: Semantic chunking failed ({str(e)[:50]}...). Using sentence fallback.[/yellow]")
                    batch_nodes = self.sentence_splitter.get_nodes_from_documents(batch)
                    self.stats["fallback_splits"] += len(batch)
                
                # Validate chunks
                validated_nodes = self._validate_chunks(batch_nodes, batch_num)
                all_nodes.extend(validated_nodes)
                
                self.stats["documents_processed"] += len(batch)
                progress.update(task, advance=len(batch))
                
                # Small delay to avoid rate limits
                if i + self.batch_size < len(documents):
                    time.sleep(0.5)
        
        self.stats["total_chunks"] = len(all_nodes)
        self.stats["total_processing_time"] = time.time() - start_time
        
        # Display stats
        self._display_stats()
        
        return all_nodes
    
    def _validate_chunks(self, nodes: List[TextNode], batch_num: int = None) -> List[TextNode]:
        """Validate chunk sizes (safety check only)
        
        Args:
            nodes: Chunks to validate
            batch_num: Batch number for logging
            
        Returns:
            Validated chunks
        """
        validated_nodes = []
        oversized_in_batch = 0
        
        for node in nodes:
            chunk_size = len(node.text)
            
            # LlamaIndex should prevent this, but safety check
            if chunk_size > self.max_chunk_chars:
                oversized_in_batch += 1
                self.stats["oversized_chunks_split"] += 1
                
                # Emergency fallback split
                doc = Document(text=node.text, metadata=node.metadata)
                split_nodes = self.sentence_splitter.get_nodes_from_documents([doc])
                validated_nodes.extend(split_nodes)
            
            # Keep small chunks as-is (merging is complex and risky)
            else:
                validated_nodes.append(node)
        
        # Log oversized chunks per batch (neutral tone)
        if oversized_in_batch > 0:
            batch_label = f"Batch {batch_num}: " if batch_num else ""
            console.print(
                f"[yellow]{batch_label}Split {oversized_in_batch} chunk(s) exceeding {self.max_chunk_chars} chars "
                f"(semantic threshold may need tuning)[/yellow]"
            )
        
        return validated_nodes
    
    def _display_stats(self):
        """Display chunking statistics"""
        from rich.table import Table
        
        table = Table(title="✨ Semantic Chunking Results", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan", width=35)
        table.add_column("Value", style="green", justify="right")
        
        # Core metrics
        table.add_row("Documents Processed", str(self.stats["documents_processed"]))
        table.add_row("Total Chunks Created", str(self.stats["total_chunks"]))
        
        # Chunk size info
        if self.stats["total_chunks"] > 0:
            avg_chunks_per_doc = self.stats["total_chunks"] / self.stats["documents_processed"]
            table.add_row("Avg Chunks per Document", f"{avg_chunks_per_doc:.1f}")
        
        table.add_row("Max Chunk Size (Enforced)", f"{self.max_chunk_chars:,} chars")
        
        # Split statistics
        if self.stats["oversized_chunks_split"] > 0:
            pct = (self.stats["oversized_chunks_split"] / self.stats["total_chunks"]) * 100
            table.add_row(
                "Chunks Requiring Split", 
                f"{self.stats['oversized_chunks_split']} ({pct:.1f}%)",
                style="yellow"
            )
        else:
            table.add_row("Chunks Requiring Split", "0 ✓", style="green")
        
        if self.stats["fallback_splits"] > 0:
            table.add_row(
                "Documents Using Fallback", 
                str(self.stats["fallback_splits"]),
                style="yellow"
            )
        
        # Performance metrics
        table.add_row("─" * 35, "─" * 15, style="dim")
        table.add_row("Total Processing Time", f"{self.stats['total_processing_time']:.2f}s")
        
        if self.stats["total_chunks"] > 0:
            avg_time = self.stats['total_processing_time'] / self.stats["total_chunks"]
            table.add_row("Avg Time per Chunk", f"{avg_time:.3f}s")
        
        if self.stats["documents_processed"] > 0:
            docs_per_sec = self.stats["documents_processed"] / self.stats["total_processing_time"]
            table.add_row("Processing Speed", f"{docs_per_sec:.1f} docs/sec")
        
        console.print("\n")
        console.print(table)
        
        # Summary message
        if self.stats["oversized_chunks_split"] == 0:
            console.print("\n[green]✓ All chunks within size limits![/green]")
        else:
            pct = (self.stats["oversized_chunks_split"] / self.stats["total_chunks"]) * 100
            console.print(
                f"\n[yellow]ℹ {pct:.1f}% of chunks exceeded limit and were split. "
                f"Consider lowering SEMANTIC_BREAKPOINT_THRESHOLD (currently {self.breakpoint_percentile_threshold}%) "
                f"for smaller chunks.[/yellow]"
            )
        
        console.print()