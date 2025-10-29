"""Health check script to verify all components"""
from rich.console import Console
from rich.table import Table
from pymilvus import connections, utility
import openai
from llama_index.vector_stores.milvus import MilvusVectorStore

from config import Config
from logger_config import setup_logger


console = Console()


def check_openai(api_key: str) -> tuple[bool, str]:
    """Check OpenAI API connectivity"""
    try:
        client = openai.OpenAI(api_key=api_key)
        # Try a simple API call
        client.models.list()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


def check_milvus(config) -> tuple[bool, str]:
    """Check Milvus server connectivity - SIMPLE VERSION"""
    try:
        # Use the exact same connection as your main app
        connections.connect(
            alias="default",
            host=config.milvus_host,
            port=config.milvus_port,
            timeout=30
        )
        
        # Verify by listing collections
        collections = utility.list_collections()
        connections.disconnect("default")
        return True, f"Connected ({len(collections)} collections)"
    except Exception as e:
        return False, str(e)


def check_documents(documents_dir) -> tuple[bool, str]:
    """Check if documents directory exists and has files"""
    try:
        if not documents_dir.exists():
            return False, "Directory not found"
        
        files = list(documents_dir.rglob('*'))
        files = [f for f in files if f.is_file()]
        
        if not files:
            return False, "No files found"
        
        return True, f"{len(files)} file(s) found"
    except Exception as e:
        return False, str(e)


def check_persistence(persist_dir) -> tuple[bool, str]:
    """Check if index persistence directory exists"""
    try:
        if not persist_dir.exists():
            return False, "Not initialized"
        
        metadata_file = persist_dir / "index_metadata.json"
        if metadata_file.exists():
            return True, "Index exists"
        else:
            return False, "No index found"
    except Exception as e:
        return False, str(e)


def main():
    """Run comprehensive health check"""
    console.print("\n[bold blue]System Health Check[/bold blue]\n")
    
    try:
        config = Config()
        logger = setup_logger('HealthCheck', config.log_dir)
        
        # Create results table
        table = Table(title="Component Status", show_header=True)
        table.add_column("Component", style="cyan", width=20)
        table.add_column("Status", width=10)
        table.add_column("Details", style="dim")
        
        # Check OpenAI
        ok, msg = check_openai(config.openai_api_key)
        table.add_row(
            "OpenAI API",
            "[green]OK[/green]" if ok else "[red]FAIL[/red]",
            msg
        )
        
        # Check Milvus Server - FIXED
        ok, msg = check_milvus(config)
        table.add_row(
            "Milvus Server",
            "[green]OK[/green]" if ok else "[red]FAIL[/red]",
            msg
        )
        
        # Check Documents
        ok, msg = check_documents(config.documents_dir)
        table.add_row(
            "Documents",
            "[green]OK[/green]" if ok else "[yellow]WARN[/yellow]",
            msg
        )
        
        # Check Persistence
        ok, msg = check_persistence(config.persist_dir)
        table.add_row(
            "Index Storage",
            "[green]OK[/green]" if ok else "[yellow]WARN[/yellow]",
            msg
        )
        
        console.print(table)
        console.print()
        
        if not ok:
            console.print("[yellow]Note: Ensure Milvus server is running with: docker-compose up -d[/yellow]")
        
        logger.info("Health check completed")
        
    except Exception as e:
        console.print(f"[red]FAIL Error during health check:[/red] {e}")


if __name__ == "__main__":
    main()