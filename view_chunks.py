from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from config import Config
from logger_config import setup_logger
from docs_processor import DocumentProcessor


console = Console()


def display_chunks():
    """Display chunks with rich formatting"""
    console.print("\n[bold blue]Document Chunks Viewer[/bold blue]\n")
    
    try:
        # Initialize
        config = Config()
        logger = setup_logger('ViewChunks', config.log_dir)
        
        console.print("[yellow]Loading existing index...[/yellow]")
        
        # Load processor with force_reindex=False to prevent reprocessing
        processor = DocumentProcessor(config, force_reindex=False)
        
        # Check if index already exists before loading
        metadata_exists = config.metadata_file.exists()
        storage_exists = (
            (config.persist_dir / "docstore.json").exists() and
            (config.persist_dir / "index_store.json").exists()
        )
        
        if not metadata_exists or not storage_exists:
            console.print("[red]No existing index found. Please run processing first.[/red]")
            console.print("Run: python cli.py process")
            return
        
        console.print(f"[dim]Index found: metadata={metadata_exists}, storage={storage_exists}[/dim]")
        
        # Load existing index without reprocessing
        processor.load_or_create_index()
        
        # Get nodes without triggering reprocessing
        nodes = processor.get_nodes()
        
        if not nodes:
            console.print("[yellow]No chunks found or could not retrieve nodes.[/yellow]")
            return
        
        # Display stats
        stats_table = Table(title="Chunk Statistics", show_header=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats = processor.get_stats()
        stats_table.add_row("Total Documents", str(stats.get('total_documents', 0)))
        stats_table.add_row("Total Chunks", str(stats.get('total_chunks', 0)))
        
        # Calculate average chunk size
        if nodes and len(nodes) > 0:
            avg_size = sum(len(n.text) for n in nodes) // len(nodes)
            stats_table.add_row("Average Chunk Size", f"{avg_size} chars")
        else:
            stats_table.add_row("Average Chunk Size", "N/A")
        
        console.print(stats_table)
        console.print()

        # After displaying chunks, add this diagnostic:
        if nodes:
            chunk_sizes = [len(node.text) for node in nodes]
            console.print(f"\n[bold]Chunk Size Analysis:[/bold]")
            console.print(f"  Min: {min(chunk_sizes)} chars")
            console.print(f"  Max: {max(chunk_sizes)} chars") 
            console.print(f"  Avg: {sum(chunk_sizes) // len(chunk_sizes)} chars")
            console.print(f"  Config: {config.chunk_size} chars (configured)")
        
        # Display chunks
        for i, node in enumerate(nodes, 1):
            # Chunk content
            content = node.text
            if len(content) > 500:
                content = content[:500] + "\n\n[dim]... (truncated, showing first 500 chars)[/dim]"
            
            console.print(Panel(content, title=f"Chunk {i}/{len(nodes)}", border_style="green"))
            console.print()
            
            # Pause every 5 chunks
            if i % 5 == 0 and i < len(nodes):
                response = console.input(f"\n[dim]Press Enter to continue ({i}/{len(nodes)} shown) or 'q' to quit: [/dim]").strip().lower()
                if response == 'q':
                    break
        
        console.print(f"\n[bold green]SUCCESS Displayed {min(i, len(nodes))}/{len(nodes)} chunks[/bold green]")
        
        # Option to save
        if console.input("\n[dim]Save full chunks to file? (y/n): [/dim]").strip().lower() == 'y':
            output_file = config.persist_dir / "chunks_full.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Total Chunks: {len(nodes)}\n")
                f.write("=" * 80 + "\n\n")
                
                for i, node in enumerate(nodes, 1):
                    f.write(f"Chunk {i}/{len(nodes)}\n")
                    f.write(f"Source: {node.metadata.get('file_name', 'Unknown')}\n")
                    f.write(f"Length: {len(node.text)} characters\n")
                    f.write(f"ID: {node.node_id if hasattr(node, 'node_id') else node.id_}\n")
                    f.write("-" * 80 + "\n")
                    f.write(node.text)
                    f.write("\n" + "=" * 80 + "\n\n")
            
            console.print(f"[green]SUCCESS[/green] Saved to {output_file}")
            logger.info(f"Saved chunks to {output_file}")
        
    except Exception as e:
        console.print(f"[red]FAIL Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    display_chunks()