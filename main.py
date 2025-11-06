# main.py
import time
from typing import List, Dict
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_exponential

from config import Config
from logger_config import setup_logger
from docs_processor import DocumentProcessor
from utils.custom_prompt import PDF_RESPONSE_PROMPT
from utils import LangSmithTracker, CostCalculator, MetricsCollector

console = Console()


class CustomPDFQueryExecutor:
    """Execute queries using the custom PDF response prompt"""
    
    def __init__(self, processor: DocumentProcessor):
        self.processor = processor
        self.llm = self._get_llm()
        self.index = processor.index
        self.logger = setup_logger('CustomPDFQueryExecutor', processor.config.log_dir)
        
        # Initialize tracking utilities
        self.langsmith_tracker = LangSmithTracker()
        self.cost_calculator = CostCalculator()
        self.metrics_collector = MetricsCollector()
    
    def _get_llm(self):
        """Get the LLM instance directly"""
        from llama_index.core import Settings
        return Settings.llm
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def execute_query(self, query: str) -> dict:
        """Execute a single query using the custom PDF prompt"""
        self.logger.info(f"Executing query with custom PDF prompt: {query}")
        
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant chunks
            retriever = self.index.as_retriever(similarity_top_k=self.processor.config.similarity_top_k)
            retrieved_nodes = retriever.retrieve(query)
            
            # Track retrieval
            chunk_data = [
                {
                    "file_name": node.metadata.get('file_name', 'Unknown'),
                    "page_label": node.metadata.get('page_label', 'N/A'),
                    "score": node.score if hasattr(node, 'score') else None,
                    "text": node.text,
                    "is_multimodal": node.metadata.get('processing_method') == 'multimodal_gpt4_vision',
                    "content_type": node.metadata.get('content_type', 'standard')
                }
                for node in retrieved_nodes
            ]
            
            self.langsmith_tracker.track_retrieval(
                query=query,
                retrieved_chunks=chunk_data,
                similarity_top_k=self.processor.config.similarity_top_k
            )
            
            # Step 2: Build context with metadata
            context = self._build_prompt_context(retrieved_nodes)
            
            # Step 3: Format the custom prompt
            final_prompt = PDF_RESPONSE_PROMPT.format(
                context=context,
                question=query
            )
            
            # Step 4: Send to LLM directly
            llm_response = self.llm.complete(final_prompt)
            
            # Calculate token usage (estimated)
            input_tokens = len(final_prompt.split()) * 1.3
            output_tokens = len(llm_response.text.split()) * 1.3
            
            # Track LLM call
            self.langsmith_tracker.track_llm_call(
                prompt=final_prompt,
                response=llm_response.text,
                model=self.processor.config.llm_model,
                temperature=self.processor.config.llm_temperature,
                metadata={
                    "chunks_used": len(retrieved_nodes),
                    "query": query
                }
            )
            
            # Calculate cost
            cost_info = self.cost_calculator.calculate_llm_cost(
                model=self.processor.config.llm_model,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens)
            )
            
            # Step 5: Process response
            sources = []
            for node in retrieved_nodes:
                source_info = {
                    'file': node.metadata.get('file_name', 'Unknown'),
                    'page': node.metadata.get('page_label', 'N/A'),
                    'score': round(node.score, 4) if hasattr(node, 'score') else None,
                    'text_preview': node.text[:150] + '...' if len(node.text) > 150 else node.text,
                    'processing_method': node.metadata.get('processing_method', 'standard'),
                    'content_type': node.metadata.get('content_type', 'standard'),
                    'is_multimodal': node.metadata.get('processing_method') == 'multimodal_gpt4_vision'
                }
                sources.append(source_info)
            
            execution_time = time.time() - start_time
            
            # Track complete query execution
            self.langsmith_tracker.track_query_execution(
                query=query,
                answer=llm_response.text,
                chunks_used=len(retrieved_nodes),
                sources=sources,
                execution_time=execution_time,
                metadata={
                    "token_info": {
                        "input_tokens": int(input_tokens),
                        "output_tokens": int(output_tokens),
                        "total_tokens": int(input_tokens + output_tokens)
                    },
                    "cost_info": cost_info
                }
            )
            
            # Record in metrics collector
            self.metrics_collector.record_query(
                query=query,
                answer=llm_response.text,
                chunks_used=len(retrieved_nodes),
                sources=sources,
                execution_time=execution_time,
                token_info={
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens)
                },
                cost_info=cost_info
            )
            
            result = {
                'query': query,
                'answer': llm_response.text,
                'sources': sources,
                'source_count': len(sources),
                'multimodal_sources': len([s for s in sources if s.get('is_multimodal')]),
                'token_info': {
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                    "total_tokens": int(input_tokens + output_tokens)
                },
                'cost_info': cost_info,
                'execution_time': execution_time
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing query: {e}", exc_info=True)
            return {
                'query': query,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _build_prompt_context(self, retrieved_nodes: List) -> str:
        """Build context for the custom prompt with proper formatting"""
        if not retrieved_nodes:
            return "No relevant documents found."
        
        context_parts = []
        for i, node in enumerate(retrieved_nodes, 1):
            filename = node.metadata.get('file_name', 'Unknown Document')
            page_number = node.metadata.get('page_label', 'N/A')
            content_type = node.metadata.get('content_type', 'text')
            is_multimodal = node.metadata.get('processing_method') == 'multimodal_gpt4_vision'
            
            # Format source header
            source_header = f"--- DOCUMENT {i}: {filename}"
            if page_number != 'N/A':
                source_header += f" (Page {page_number})"
            if is_multimodal:
                source_header += f" [Multi-modal: {content_type}]"
            source_header += " ---"
            
            context_parts.append(source_header)
            context_parts.append("")
            
            # Add text content
            context_parts.append(node.text)
            context_parts.append("")
            context_parts.append("=" * 80)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def execute_batch(self, queries: List[str]):
        """Execute multiple queries with custom PDF prompt"""
        results = []
        
        console.print(f"\n[bold blue]Executing {len(queries)} queries...[/bold blue]\n")
        
        for i, query in enumerate(queries, 1):
            console.print(f"[bold cyan]Query {i}/{len(queries)}[/bold cyan]")
            console.print(Panel(query, title="Question", border_style="cyan"))
            
            try:
                result = self.execute_query(query)
                results.append(result)
                
                if 'error' not in result:
                    self._display_pdf_result(result)
                else:
                    console.print(f"[red]❌ Error: {result['error']}[/red]")
                    
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")
                results.append({'query': query, 'error': str(e)})
        
        # Display session summary
        console.print("\n[bold green]📊 Session Summary[/bold green]")
        self.metrics_collector.display_summary()
        self.cost_calculator.display_session_costs()
        
        # Export metrics
        metrics_dir = Path("./logs/metrics")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_collector.export_metrics(metrics_dir / f"metrics_{timestamp}.json")
        
        return results
    
    def _display_pdf_result(self, result: Dict):
        """Display result from custom PDF prompt"""
        # Show processing method
        multimodal_count = result.get('multimodal_sources', 0)
        if multimodal_count > 0:
            console.print(f"[dim]🎨 Using {multimodal_count} multi-modal source(s)[/dim]")
        
        # Display the answer
        console.print(Panel(
            Markdown(result['answer']),
            title="Answer",
            border_style="green"
        ))
        
        # Display sources
        if result['sources']:
            console.print(f"\n[bold]📚 Sources Used ({len(result['sources'])}):[/bold]")
            for i, source in enumerate(result['sources'][:3], 1):
                source_name = source['file']
                if source['page'] != 'N/A':
                    source_name += f" (Page {source['page']})"
                
                # Add processing method badge
                if source.get('is_multimodal'):
                    content_type = source.get('content_type', 'image')
                    badge = f" [🎨 {content_type.title()}]"
                else:
                    badge = ""
                
                console.print(f"  {i}. {source_name}{badge}")
                
                if source.get('score'):
                    console.print(f"     Relevance: {source['score']:.3f}")


def check_processing_status(processor: DocumentProcessor):
    """Check and display document processing status"""
    console.print("\n[bold blue]🔍 Checking Document Processing Status...[/bold blue]")
    
    try:
        # Get document statistics
        doc_stats = processor.get_document_statistics()
        total_docs = doc_stats.get('total_documents', 0)
        multimodal_docs = doc_stats.get('multimodal_documents', 0)
        standard_docs = doc_stats.get('standard_documents', 0)
        
        console.print(f"[dim]📊 Documents Processed: {total_docs} total")
        console.print(f"[dim]   - Multi-Modal (GPT-4 Vision): {multimodal_docs} documents")
        console.print(f"[dim]   - Standard: {standard_docs} documents")
        
        # Multi-modal stats
        if multimodal_docs > 0:
            console.print(f"\n[green]✅ Multi-modal processing (GPT-4 Vision)[/green]")
            console.print(f"[dim]   Tables: {doc_stats.get('multimodal_tables', 0)}")
            console.print(f"[dim]   Charts: {doc_stats.get('multimodal_charts', 0)}")
            console.print(f"[dim]   Text: {doc_stats.get('multimodal_text', 0)}")
            console.print(f"[dim]   Mixed: {doc_stats.get('multimodal_mixed', 0)}")
            console.print(f"[dim]   Total Words: {doc_stats.get('total_words_multimodal', 0)}")
            console.print(f"[dim]   Avg Confidence: {doc_stats.get('avg_confidence_multimodal', 0):.1f}%")
        
        console.print()
        
    except Exception as e:
        console.print(f"[yellow]⚠ Could not retrieve processing status: {e}[/yellow]")


def main():
    """Main query execution"""
    console.print("\n[bold blue]📚 Document Query System[/bold blue]\n")
    console.print("[dim]Enhanced with multi-modal processing (GPT-4 Vision)[/dim]\n")
    
    try:
        # Initialize
        config = Config()
        logger = setup_logger('Main', config.log_dir)
        
        logger.info("Initializing query system...")
        console.print("[yellow]Initializing query engine...[/yellow]")
        
        # Load processor
        processor = DocumentProcessor(config, force_reindex=False)
        
        # Check if we have an existing index
        metadata_exists = config.metadata_file.exists()
        storage_exists = (
            (config.persist_dir / "docstore.json").exists() and
            (config.persist_dir / "index_store.json").exists()
        )
        
        if metadata_exists and storage_exists:
            console.print("[green]✓[/green] Existing index found, loading...")
            processor.load_or_create_index()
        else:
            console.print("[yellow]No existing index found, creating new one...[/yellow]")
            processor.load_or_create_index()
        
        # Check and display processing status
        check_processing_status(processor)
        
        # Initialize query executor
        executor = CustomPDFQueryExecutor(processor)
        console.print("[green]✓ Query engine ready[/green]\n")
        
        # Interactive mode
        console.print("[bold]Interactive mode.[/bold] Type 'exit' or 'quit' to leave.")
        while True:
            console.print("\n[bold]Enter your queries (one per line). Press Enter on an empty line to run.[/bold]")
            queries: List[str] = []
            while True:
                line = console.input("[cyan]Query: [/cyan]").strip()
                if not line:
                    break
                queries.append(line)
            
            if not queries:
                console.print("[yellow]No queries entered.[/yellow]")
                continue
            
            # Check for exit command
            if any(q.lower() in ['exit', 'quit'] for q in queries):
                console.print("[yellow]Exiting...[/yellow]")
                break
            
            # Execute queries
            results = executor.execute_batch(queries)
            
            # Summary
            successful = len([r for r in results if 'error' not in r])
            total_multimodal = sum(r.get('multimodal_sources', 0) for r in results if 'error' not in r)
            
            console.print(f"\n[bold green]✓ Execution Complete: {successful}/{len(queries)} queries successful[/bold green]")
            
            if total_multimodal > 0:
                console.print(f"[cyan]🎨 Used {total_multimodal} multi-modal source(s) (GPT-4 Vision)[/cyan]")
            
            logger.info(f"Query execution completed: {successful}/{len(queries)} successful")
        
    except ValueError as e:
        console.print(f"[red]✗ Configuration Error:[/red] {e}")
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    main()