"""LangSmith integration for tracking LLM calls and metrics"""
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
from langsmith import Client
from langsmith.run_helpers import traceable
from rich.console import Console

console = Console()


class LangSmithTracker:
    """Track LLM interactions using LangSmith"""
    
    def __init__(self, project_name: str = "llamaindex-document-query"):
        """Initialize LangSmith tracker
        
        Args:
            project_name: Name of the LangSmith project
        """
        self.project_name = project_name
        self.client = None
        self.enabled = False
        
        # Try to initialize LangSmith client
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize LangSmith client with API key from environment"""
        api_key = os.getenv("LANGSMITH_API_KEY")
        
        if not api_key:
            console.print("[yellow]⚠ LANGSMITH_API_KEY not found. LangSmith tracking disabled.[/yellow]")
            console.print("[dim]To enable: Set LANGSMITH_API_KEY in .env file[/dim]")
            return
        
        try:
            self.client = Client(api_key=api_key)
            self.enabled = True
            console.print(f"[green]✓[/green] LangSmith tracking enabled for project: {self.project_name}")
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to initialize LangSmith: {e}[/yellow]")
            self.enabled = False
    
    @traceable(run_type="llm")
    def track_llm_call(
        self,
        prompt: str,
        response: str,
        model: str,
        temperature: float,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Track an LLM call with LangSmith
        
        Args:
            prompt: The input prompt
            response: The LLM response
            model: Model name (e.g., 'gpt-4o')
            temperature: Temperature setting
            metadata: Additional metadata to track
            
        Returns:
            Dictionary with tracking info
        """
        if not self.enabled:
            return {}
        
        tracking_data = {
            "prompt": prompt,
            "response": response,
            "model": model,
            "temperature": temperature,
            "timestamp": datetime.now().isoformat(),
            "project": self.project_name,
            **(metadata or {})
        }
        
        return tracking_data
    
    @traceable(run_type="retriever")
    def track_retrieval(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        similarity_top_k: int,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Track chunk retrieval operation
        
        Args:
            query: The query string
            retrieved_chunks: List of retrieved chunks with metadata
            similarity_top_k: Number of chunks requested
            metadata: Additional metadata
            
        Returns:
            Dictionary with retrieval tracking info
        """
        if not self.enabled:
            return {}
        
        tracking_data = {
            "query": query,
            "num_chunks_requested": similarity_top_k,
            "num_chunks_retrieved": len(retrieved_chunks),
            "chunk_details": [
                {
                    "file": chunk.get("file_name", "unknown"),
                    "page": chunk.get("page_label", "N/A"),
                    "score": chunk.get("score"),
                    "length": len(chunk.get("text", "")),
                    "is_ocr": chunk.get("is_ocr_processed", False)
                }
                for chunk in retrieved_chunks
            ],
            "timestamp": datetime.now().isoformat(),
            "project": self.project_name,
            **(metadata or {})
        }
        
        return tracking_data
    
    @traceable(run_type="chain")
    def track_query_execution(
        self,
        query: str,
        answer: str,
        chunks_used: int,
        sources: List[Dict],
        execution_time: float,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Track complete query execution
        
        Args:
            query: The user query
            answer: The generated answer
            chunks_used: Number of chunks used
            sources: List of source documents
            execution_time: Time taken in seconds
            metadata: Additional metadata (tokens, cost, etc.)
            
        Returns:
            Dictionary with execution tracking info
        """
        if not self.enabled:
            return {}
        
        tracking_data = {
            "query": query,
            "answer": answer,
            "chunks_used": chunks_used,
            "sources": sources,
            "execution_time_seconds": execution_time,
            "timestamp": datetime.now().isoformat(),
            "project": self.project_name,
            **(metadata or {})
        }
        
        return tracking_data
    
    def get_project_stats(self) -> Optional[Dict]:
        """Get statistics for the current project
        
        Returns:
            Dictionary with project statistics or None if not enabled
        """
        if not self.enabled or not self.client:
            return None
        
        try:
            # Get runs for this project
            runs = list(self.client.list_runs(project_name=self.project_name))
            
            total_runs = len(runs)
            llm_runs = len([r for r in runs if r.run_type == "llm"])
            retrieval_runs = len([r for r in runs if r.run_type == "retriever"])
            chain_runs = len([r for r in runs if r.run_type == "chain"])
            
            return {
                "project_name": self.project_name,
                "total_runs": total_runs,
                "llm_calls": llm_runs,
                "retrieval_operations": retrieval_runs,
                "query_executions": chain_runs,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to get project stats: {e}[/yellow]")
            return None
    
    def export_session_data(self, output_file: Path):
        """Export tracking data to JSON file
        
        Args:
            output_file: Path to output JSON file
        """
        if not self.enabled:
            console.print("[yellow]LangSmith not enabled, cannot export data[/yellow]")
            return
        
        stats = self.get_project_stats()
        if stats:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=2)
            console.print(f"[green]✓[/green] Exported LangSmith data to {output_file}")