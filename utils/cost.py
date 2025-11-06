"""Calculate OpenAI API costs based on token usage"""
from typing import Dict, Optional
from rich.console import Console
from rich.table import Table

console = Console()


class CostCalculator:
    """Calculate costs for OpenAI API usage"""
    
    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        # GPT-4 models
        "gpt-4o": {
            "input": 2.50,   # $2.50 per 1M input tokens
            "output": 10.00  # $10.00 per 1M output tokens
        },
        "gpt-4o-mini": {
            "input": 0.150,
            "output": 0.600
        },
        "gpt-4-turbo": {
            "input": 10.00,
            "output": 30.00
        },
        "gpt-4": {
            "input": 30.00,
            "output": 60.00
        },
        
        # GPT-3.5 models
        "gpt-3.5-turbo": {
            "input": 0.50,
            "output": 1.50
        },
        
        # Embeddings
        "text-embedding-3-small": {
            "input": 0.020,
            "output": 0.0
        },
        "text-embedding-3-large": {
            "input": 0.130,
            "output": 0.0
        },
        "text-embedding-ada-002": {
            "input": 0.100,
            "output": 0.0
        }
    }
    
    def __init__(self):
        """Initialize cost calculator"""
        self.session_costs = {
            "llm_calls": [],
            "embedding_calls": [],
            "total_cost": 0.0
        }
    
    def calculate_llm_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, float]:
        """Calculate cost for an LLM call
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Dictionary with cost breakdown
        """
        if model not in self.PRICING:
            console.print(f"[yellow]⚠ Unknown model '{model}', using gpt-4o pricing[/yellow]")
            model = "gpt-4o"
        
        pricing = self.PRICING[model]
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        cost_data = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
        
        # Track in session
        self.session_costs["llm_calls"].append(cost_data)
        self.session_costs["total_cost"] += total_cost
        
        return cost_data
    
    def calculate_embedding_cost(
        self,
        model: str,
        tokens: int
    ) -> Dict[str, float]:
        """Calculate cost for embedding generation
        
        Args:
            model: Embedding model name
            tokens: Number of tokens
            
        Returns:
            Dictionary with cost info
        """
        if model not in self.PRICING:
            console.print(f"[yellow]⚠ Unknown embedding model '{model}', using text-embedding-3-small pricing[/yellow]")
            model = "text-embedding-3-small"
        
        pricing = self.PRICING[model]
        cost = (tokens / 1_000_000) * pricing["input"]
        
        cost_data = {
            "model": model,
            "tokens": tokens,
            "cost": cost
        }
        
        # Track in session
        self.session_costs["embedding_calls"].append(cost_data)
        self.session_costs["total_cost"] += cost
        
        return cost_data
    
    def get_session_summary(self) -> Dict:
        """Get summary of session costs
        
        Returns:
            Dictionary with session cost summary
        """
        total_llm_calls = len(self.session_costs["llm_calls"])
        total_embedding_calls = len(self.session_costs["embedding_calls"])
        
        total_input_tokens = sum(call["input_tokens"] for call in self.session_costs["llm_calls"])
        total_output_tokens = sum(call["output_tokens"] for call in self.session_costs["llm_calls"])
        total_embedding_tokens = sum(call["tokens"] for call in self.session_costs["embedding_calls"])
        
        return {
            "llm_calls": total_llm_calls,
            "embedding_calls": total_embedding_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_embedding_tokens": total_embedding_tokens,
            "total_cost": self.session_costs["total_cost"]
        }
    
    def display_session_costs(self):
        """Display session costs in a rich table"""
        summary = self.get_session_summary()
        
        table = Table(title="Session Cost Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("LLM Calls", str(summary["llm_calls"]))
        table.add_row("Embedding Calls", str(summary["embedding_calls"]))
        table.add_row("Total Input Tokens", f"{summary['total_input_tokens']:,}")
        table.add_row("Total Output Tokens", f"{summary['total_output_tokens']:,}")
        table.add_row("Total Embedding Tokens", f"{summary['total_embedding_tokens']:,}")
        table.add_row(
            "Total Cost",
            f"${summary['total_cost']:.4f}",
            style="bold green"
        )
        
        console.print("\n")
        console.print(table)
    
    def reset_session(self):
        """Reset session tracking"""
        self.session_costs = {
            "llm_calls": [],
            "embedding_calls": [],
            "total_cost": 0.0
        }