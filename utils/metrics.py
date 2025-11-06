"""Collect and aggregate metrics for query operations"""
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table

console = Console()


class MetricsCollector:
    """Collect and aggregate metrics for queries and operations"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics = {
            "queries": [],
            "total_queries": 0,
            "total_chunks_used": 0,
            "total_execution_time": 0.0,
            "ocr_sources_used": 0,
            "session_start": datetime.now().isoformat()
        }
    
    def record_query(
        self,
        query: str,
        answer: str,
        chunks_used: int,
        sources: List[Dict],
        execution_time: float,
        token_info: Optional[Dict] = None,
        cost_info: Optional[Dict] = None
    ):
        """Record a query execution with all metrics
        
        Args:
            query: The query string
            answer: Generated answer
            chunks_used: Number of chunks used
            sources: List of source documents
            execution_time: Execution time in seconds
            token_info: Token usage information
            cost_info: Cost information
        """
        ocr_sources = len([s for s in sources if s.get("is_ocr_processed")])
        
        query_record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer_length": len(answer),
            "chunks_used": chunks_used,
            "sources_count": len(sources),
            "ocr_sources": ocr_sources,
            "execution_time": execution_time,
            "token_info": token_info or {},
            "cost_info": cost_info or {}
        }
        
        self.metrics["queries"].append(query_record)
        self.metrics["total_queries"] += 1
        self.metrics["total_chunks_used"] += chunks_used
        self.metrics["total_execution_time"] += execution_time
        self.metrics["ocr_sources_used"] += ocr_sources
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics
        
        Returns:
            Dictionary with aggregated metrics
        """
        if self.metrics["total_queries"] == 0:
            return {
                "total_queries": 0,
                "message": "No queries recorded yet"
            }
        
        avg_chunks = self.metrics["total_chunks_used"] / self.metrics["total_queries"]
        avg_time = self.metrics["total_execution_time"] / self.metrics["total_queries"]
        
        # Calculate token totals
        total_input_tokens = sum(
            q.get("token_info", {}).get("input_tokens", 0)
            for q in self.metrics["queries"]
        )
        total_output_tokens = sum(
            q.get("token_info", {}).get("output_tokens", 0)
            for q in self.metrics["queries"]
        )
        
        # Calculate cost totals
        total_cost = sum(
            q.get("cost_info", {}).get("total_cost", 0)
            for q in self.metrics["queries"]
        )
        
        return {
            "session_start": self.metrics["session_start"],
            "total_queries": self.metrics["total_queries"],
            "total_chunks_used": self.metrics["total_chunks_used"],
            "avg_chunks_per_query": avg_chunks,
            "total_execution_time": self.metrics["total_execution_time"],
            "avg_execution_time": avg_time,
            "ocr_sources_used": self.metrics["ocr_sources_used"],
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_cost": total_cost
        }
    
    def display_summary(self):
        """Display metrics summary in a rich table"""
        summary = self.get_summary()
        
        if "message" in summary:
            console.print(f"[yellow]{summary['message']}[/yellow]")
            return
        
        table = Table(title="Query Metrics Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Queries", str(summary["total_queries"]))
        table.add_row("Total Chunks Used", str(summary["total_chunks_used"]))
        table.add_row("Avg Chunks/Query", f"{summary['avg_chunks_per_query']:.1f}")
        table.add_row("Total Execution Time", f"{summary['total_execution_time']:.2f}s")
        table.add_row("Avg Execution Time", f"{summary['avg_execution_time']:.2f}s")
        table.add_row("OCR Sources Used", str(summary["ocr_sources_used"]))
        table.add_row("Total Input Tokens", f"{summary['total_input_tokens']:,}")
        table.add_row("Total Output Tokens", f"{summary['total_output_tokens']:,}")
        table.add_row("Total Tokens", f"{summary['total_tokens']:,}")
        table.add_row("Total Cost", f"${summary['total_cost']:.4f}", style="bold green")
        
        console.print("\n")
        console.print(table)
    
    def export_metrics(self, output_file: Path):
        """Export metrics to JSON file
        
        Args:
            output_file: Path to output file
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            **self.metrics,
            "summary": self.get_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        console.print(f"[green]✓[/green] Exported metrics to {output_file}")
    
    def get_query_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get query history
        
        Args:
            limit: Maximum number of queries to return (None for all)
            
        Returns:
            List of query records
        """
        queries = self.metrics["queries"]
        if limit:
            return queries[-limit:]
        return queries