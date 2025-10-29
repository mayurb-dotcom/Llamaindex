from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from tenacity import retry, stop_after_attempt, wait_exponential

from config import Config
from logger_config import setup_logger
from docs_processor import DocumentProcessor


console = Console()


class QueryExecutor:
    """Execute queries with error handling and retry logic"""
    
    def __init__(self, processor: DocumentProcessor):
        self.processor = processor
        self.query_engine = processor.get_query_engine()
        self.logger = setup_logger('QueryExecutor', processor.config.log_dir)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def execute_query(self, query: str) -> dict:
        """Execute a single query with retry logic"""
        self.logger.info(f"Executing query: {query}")
        
        try:
            response = self.query_engine.query(query)
            
            result = {
                'query': query,
                'answer': response.response,
                'sources': []
            }
            
            # Extract source information
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    result['sources'].append({
                        'file': node.metadata.get('file_name', 'Unknown'),
                        'score': round(node.score, 4) if hasattr(node, 'score') else None,
                        'text_preview': node.text[:150] + '...' if len(node.text) > 150 else node.text
                    })
            
            self.logger.info(f"Query executed successfully with {len(result['sources'])} sources")
            return result
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}", exc_info=True)
            raise
    
    def execute_batch(self, queries: List[str]):
        """Execute multiple queries"""
        results = []
        
        console.print(f"\n[bold blue]Executing {len(queries)} queries...[/bold blue]\n")
        
        for i, query in enumerate(queries, 1):
            console.print(f"[bold cyan]Query {i}/{len(queries)}[/bold cyan]")
            console.print(Panel(query, title="Question", border_style="cyan"))
            
            try:
                result = self.execute_query(query)
                results.append(result)
                
                # Display answer
                console.print(Panel(
                    Markdown(result['answer']),
                    title="Answer",
                    border_style="green"
                ))
                
                # Display sources
                if result['sources']:
                    console.print(f"\n[dim]Sources ({len(result['sources'])} chunks):[/dim]")
                    for j, source in enumerate(result['sources'][:3], 1):  # Show top 3
                        score_str = f" (similarity: {source['score']})" if source['score'] else ""
                        console.print(f"  {j}. {source['file']}{score_str}")
                
                console.print()
                
            except Exception as e:
                console.print(f"[red]✗ Query failed: {e}[/red]\n")
                self.logger.error(f"Query {i} failed: {e}")
                results.append({
                    'query': query,
                    'error': str(e)
                })
        
        return results


def main():
    """Main query execution with robust error handling - FIXED VERSION"""
    console.print("\n[bold blue]LlamaIndex Query System[/bold blue]\n")
    
    try:
        # Initialize
        config = Config()
        logger = setup_logger('Main', config.log_dir)
        
        logger.info("Initializing query system...")
        console.print("[yellow]Initializing query engine...[/yellow]")
        
        # Load processor with force_reindex=False to prevent reprocessing
        processor = DocumentProcessor(config, force_reindex=False)
        
        # Check if we have an existing index first
        metadata_exists = config.metadata_file.exists()
        storage_exists = (
            (config.persist_dir / "docstore.json").exists() and
            (config.persist_dir / "index_store.json").exists()
        )
        
        if metadata_exists and storage_exists:
            console.print("[green]✓[/green] Existing index found, loading...")
            processor.load_or_create_index()  # This will only process if changes detected
        else:
            console.print("[yellow]No existing index found, creating new one...[/yellow]")
            # Only use streaming for truly large document sets
            estimated_docs = processor.get_document_count()
            if config.enable_streaming and estimated_docs > config.large_document_threshold:
                processor.process_large_document_set()
            else:
                processor.load_or_create_index()
        
        # Create query executor
        executor = QueryExecutor(processor)
        
        console.print("[green]✓[/green] Query engine ready\n")
        
        # Define queries
        queries = [
            "What is the main purpose of the Recruitment Policy at The Great Eastern Shipping Co. Ltd.?",
            "What stages are involved in the recruitment process at GE Shipping?",
            "What objectives are outlined under Manpower Planning in the recruitment process?",
            "Which HR process requires consultation with Heads of Department (HODs) and Executive Directors (EDs)?",
            "How does GE Shipping define the job specifications for recruitment?",
            "What sources does the HR team at GE Shipping use to identify potential candidates?",
            "What type of test is mandatory for Senior Managers during the selection process?",
            "What additional checks are included in the reference verification process?",
            "Where should candidates residing outside Mumbai complete their pre-employment medical tests?",
            "What is the aim of the induction and onboarding program for new recruits?",
            "How long is the probation period for new employees at GE Shipping?",
            "What evaluation forms are sent out prior to confirming an employee post-probation?",
            "What types of employees are required to adhere to the 9-hour workday schedule?",
            "What is the maximum daily meal reimbursement for employees on outdoor duty?",
            "Which department oversees the appraisal process for fleet staff?",
            "How does the Performance Management system benefit both managers and employees?",
            "What are the key focus areas of the Compensation and Rewards system?",
            "What options are available under Performance Incentive Pay?",
            "What benefits are associated with the Leave Without Pay policy?",
            "What training opportunities are provided to employees for career development?",
        ]
        
        # Execute queries
        results = executor.execute_batch(queries)
        
        # Summary
        successful = len([r for r in results if 'error' not in r])
        console.print(f"\n[bold green]✓ Completed: {successful}/{len(queries)} queries successful[/bold green]")
        
        logger.info(f"Query execution completed: {successful}/{len(queries)} successful")
        
    except ValueError as e:
        console.print(f"[red]✗ Configuration Error:[/red] {e}")
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    main()
