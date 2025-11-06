"""View and analyze document chunks from the index"""
from pathlib import Path
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from llama_index.core.schema import TextNode

from config import Config
from docs_processor import DocumentProcessor
from logger_config import setup_logger

console = Console()


def display_chunk_analysis(nodes: List[TextNode]):
    """Display detailed chunk analysis"""
    
    # Statistics
    stats_table = Table(title="📊 Chunk Statistics", box=box.ROUNDED, show_header=True)
    stats_table.add_column("Metric", style="cyan", width=30)
    stats_table.add_column("Value", style="green", justify="right")
    
    chunk_sizes = [len(node.text) for node in nodes]
    
    stats_table.add_row("Total Chunks", f"{len(nodes):,}")
    stats_table.add_row("Average Chunk Size", f"{sum(chunk_sizes) // len(nodes):,} chars")
    stats_table.add_row("Minimum Chunk Size", f"{min(chunk_sizes):,} chars")
    stats_table.add_row("Maximum Chunk Size", f"{max(chunk_sizes):,} chars")
    
    # Document distribution
    docs = {}
    for node in nodes:
        doc_name = node.metadata.get('file_name', 'Unknown')
        docs[doc_name] = docs.get(doc_name, 0) + 1
    
    stats_table.add_row("Unique Documents", str(len(docs)))
    
    # Processing methods
    processing_methods = {}
    for node in nodes:
        method = node.metadata.get('processing_method', 'standard')
        processing_methods[method] = processing_methods.get(method, 0) + 1
    
    for method, count in processing_methods.items():
        stats_table.add_row(f"  {method.capitalize()} Chunks", f"{count:,}")
    
    console.print("\n")
    console.print(stats_table)
    console.print()
    
    # Size distribution
    size_ranges = {
        "0-500": 0,
        "501-1000": 0,
        "1001-1500": 0,
        "1501-2000": 0,
        "2000+": 0
    }
    
    for size in chunk_sizes:
        if size <= 500:
            size_ranges["0-500"] += 1
        elif size <= 1000:
            size_ranges["501-1000"] += 1
        elif size <= 1500:
            size_ranges["1001-1500"] += 1
        elif size <= 2000:
            size_ranges["1501-2000"] += 1
        else:
            size_ranges["2000+"] += 1
    
    dist_table = Table(title="📏 Chunk Size Distribution", box=box.ROUNDED)
    dist_table.add_column("Size Range (chars)", style="cyan")
    dist_table.add_column("Count", style="green", justify="right")
    dist_table.add_column("Percentage", style="yellow", justify="right")
    
    for range_name, count in size_ranges.items():
        pct = (count / len(nodes)) * 100
        dist_table.add_row(range_name, f"{count:,}", f"{pct:.1f}%")
    
    console.print(dist_table)
    console.print()
    
    # Top documents by chunk count
    top_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)[:10]
    
    doc_table = Table(title="📄 Top 10 Documents by Chunk Count", box=box.ROUNDED)
    doc_table.add_column("Document", style="cyan", width=50)
    doc_table.add_column("Chunks", style="green", justify="right")
    
    for doc_name, chunk_count in top_docs:
        display_name = doc_name if len(doc_name) <= 50 else doc_name[:47] + "..."
        doc_table.add_row(display_name, str(chunk_count))
    
    console.print(doc_table)
    console.print()


def display_sample_chunks(nodes: List[TextNode], num_samples: int = 5):
    """Display sample chunks"""
    console.print(f"\n[bold cyan]📝 Sample Chunks (showing {num_samples} of {len(nodes)})[/bold cyan]\n")
    
    # Show diverse samples (different documents, different sizes)
    import random
    
    # Group by document
    by_doc = {}
    for node in nodes:
        doc = node.metadata.get('file_name', 'Unknown')
        if doc not in by_doc:
            by_doc[doc] = []
        by_doc[doc].append(node)
    
    # Sample from different documents
    samples = []
    doc_list = list(by_doc.keys())
    random.shuffle(doc_list)
    
    for doc in doc_list[:num_samples]:
        if by_doc[doc]:
            samples.append(random.choice(by_doc[doc]))
    
    # If we need more samples, add random ones
    while len(samples) < num_samples and len(samples) < len(nodes):
        node = random.choice(nodes)
        if node not in samples:
            samples.append(node)
    
    for i, node in enumerate(samples, 1):
        # Metadata
        metadata_lines = []
        metadata_lines.append(f"[bold]Document:[/bold] {node.metadata.get('file_name', 'Unknown')}")
        
        if 'page_label' in node.metadata:
            metadata_lines.append(f"[bold]Page:[/bold] {node.metadata['page_label']}")
        
        metadata_lines.append(f"[bold]Length:[/bold] {len(node.text)} chars")
        
        if 'processing_method' in node.metadata:
            metadata_lines.append(f"[bold]Method:[/bold] {node.metadata['processing_method']}")
        
        if 'content_type' in node.metadata:
            metadata_lines.append(f"[bold]Type:[/bold] {node.metadata['content_type']}")
        
        metadata_info = " | ".join(metadata_lines)
        
        # Content preview
        content = node.text
        if len(content) > 600:
            content = content[:600] + f"\n\n[dim]... (truncated, showing first 600 chars of {len(node.text)})[/dim]"
        
        console.print(Panel(
            content,
            title=f"Sample {i}/{num_samples} - {metadata_info}",
            border_style="blue",
            box=box.ROUNDED
        ))
        console.print()


def search_chunks(nodes: List[TextNode]):
    """Interactive chunk search"""
    console.print("\n[bold cyan]🔍 Interactive Chunk Search[/bold cyan]\n")
    
    while True:
        query = console.input("[yellow]Enter search term (or 'q' to quit): [/yellow]").strip()
        
        if query.lower() == 'q':
            break
        
        if not query:
            continue
        
        # Search in chunk text
        matches = []
        for node in nodes:
            if query.lower() in node.text.lower():
                matches.append(node)
        
        if not matches:
            console.print(f"[red]No chunks found containing '{query}'[/red]\n")
            continue
        
        console.print(f"\n[green]✓ Found {len(matches)} chunk(s) containing '{query}'[/green]\n")
        
        # Show first 5 matches
        for i, node in enumerate(matches[:5], 1):
            # Find context around match
            text_lower = node.text.lower()
            query_lower = query.lower()
            pos = text_lower.find(query_lower)
            
            start = max(0, pos - 100)
            end = min(len(node.text), pos + len(query) + 100)
            snippet = node.text[start:end]
            
            if start > 0:
                snippet = "..." + snippet
            if end < len(node.text):
                snippet = snippet + "..."
            
            # Highlight match (approximate, case-insensitive)
            snippet = snippet.replace(query, f"[bold yellow]{query}[/bold yellow]")
            snippet = snippet.replace(query.lower(), f"[bold yellow]{query.lower()}[/bold yellow]")
            snippet = snippet.replace(query.upper(), f"[bold yellow]{query.upper()}[/bold yellow]")
            snippet = snippet.replace(query.capitalize(), f"[bold yellow]{query.capitalize()}[/bold yellow]")
            
            metadata_info = f"{node.metadata.get('file_name', 'Unknown')}"
            if 'page_label' in node.metadata:
                metadata_info += f" (Page {node.metadata['page_label']})"
            
            console.print(Panel(
                snippet,
                title=f"Match {i}/{min(5, len(matches))} - {metadata_info}",
                border_style="green",
                box=box.SIMPLE
            ))
        
        if len(matches) > 5:
            console.print(f"\n[dim]({len(matches) - 5} more matches not shown)[/dim]\n")
        else:
            console.print()


def save_chunks_to_file(nodes: List[TextNode], output_file: Path):
    """Save all chunks to a text file"""
    console.print(f"\n[yellow]Saving {len(nodes)} chunks to {output_file}...[/yellow]")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Total Chunks: {len(nodes)}\n")
        f.write(f"Generated: {Path(__file__).name}\n")
        f.write("=" * 100 + "\n\n")
        
        for i, node in enumerate(nodes, 1):
            f.write(f"CHUNK {i}/{len(nodes)}\n")
            f.write(f"{'─' * 100}\n")
            f.write(f"Source: {node.metadata.get('file_name', 'Unknown')}\n")
            
            if 'page_label' in node.metadata:
                f.write(f"Page: {node.metadata['page_label']}\n")
            
            f.write(f"Length: {len(node.text)} characters\n")
            f.write(f"Node ID: {node.node_id if hasattr(node, 'node_id') else node.id_}\n")
            
            if 'processing_method' in node.metadata:
                f.write(f"Processing: {node.metadata['processing_method']}\n")
            
            f.write(f"{'─' * 100}\n\n")
            f.write(node.text)
            f.write("\n\n" + "=" * 100 + "\n\n")
    
    console.print(f"[green]✓ Saved to {output_file}[/green]")


def display_chunks():
    """Main function to display chunks with rich formatting"""
    console.print("\n[bold blue]═══════════════════════════════════════[/bold blue]")
    console.print("[bold blue]       Document Chunks Viewer[/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════[/bold blue]\n")
    
    try:
        # Initialize
        config = Config()
        logger = setup_logger('ViewChunks', config.log_dir)
        
        console.print("[yellow]Loading existing index...[/yellow]")
        
        # Load processor
        processor = DocumentProcessor(config, force_reindex=False)
        processor.load_or_create_index()
        
        # Get nodes
        nodes = processor.get_nodes()
        
        if not nodes:
            console.print("\n[red]✗ No chunks found in index[/red]")
            return
        
        console.print(f"[green]✓ Loaded {len(nodes)} chunks successfully[/green]\n")
        
        # Main menu loop
        while True:
            console.print("\n[bold cyan]━━━ Options ━━━[/bold cyan]")
            console.print("1. View chunk statistics")
            console.print("2. View sample chunks")
            console.print("3. Search chunks")
            console.print("4. Save all chunks to file")
            console.print("5. Exit")
            
            choice = console.input("\n[yellow]Select option (1-5): [/yellow]").strip()
            
            if choice == '1':
                display_chunk_analysis(nodes)
            
            elif choice == '2':
                num = console.input("[yellow]How many samples to show? (default 5): [/yellow]").strip()
                num_samples = int(num) if num.isdigit() else 5
                display_sample_chunks(nodes, num_samples)
            
            elif choice == '3':
                search_chunks(nodes)
            
            elif choice == '4':
                output_file = config.persist_dir / "chunks_export.txt"
                save_chunks_to_file(nodes, output_file)
            
            elif choice == '5':
                console.print("\n[green]Goodbye! 👋[/green]\n")
                break
            
            else:
                console.print("[red]Invalid option, please try again[/red]")
        
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    display_chunks()