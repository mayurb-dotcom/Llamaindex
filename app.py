"""
FastAPI web application for document query system
"""
import asyncio
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from config import Config
from logger_config import setup_logger
from docs_processor import DocumentProcessor
from main import CustomPDFQueryExecutor

# Initialize FastAPI app
app = FastAPI(
    title="LlamaIndex",
    description="Query your documents using LlamaIndex and GPT-4",
    version="1.0.0"
)

# Global state
processor: Optional[DocumentProcessor] = None
executor: Optional[CustomPDFQueryExecutor] = None
logger = None

# Templates directory
templates = Jinja2Templates(directory="templates")


class QueryRequest(BaseModel):
    """Request model for queries"""
    query: str
    top_k: Optional[int] = 15


class QueryResponse(BaseModel):
    """Response model for queries"""
    query: str
    answer: str
    sources: List[Dict]
    execution_time: float
    multimodal_sources: int
    token_info: Dict
    cost_info: Dict


@app.on_event("startup")
async def startup_event():
    """Initialize the query system on startup"""
    global processor, executor, logger
    
    try:
        config = Config()
        logger = setup_logger('FastAPI', config.log_dir)
        logger.info("Initializing FastAPI application...")
        
        # Load processor
        processor = DocumentProcessor(config, force_reindex=False)
        
        # Check if index exists
        metadata_exists = config.metadata_file.exists()
        storage_exists = (
            (config.persist_dir / "docstore.json").exists() and
            (config.persist_dir / "index_store.json").exists()
        )
        
        if metadata_exists and storage_exists:
            logger.info("Loading existing index...")
            processor.load_or_create_index()
        else:
            logger.warning("No existing index found. Please run docs_processor.py first.")
            raise RuntimeError("No document index found. Run docs_processor.py to create one.")
        
        # Initialize query executor
        executor = CustomPDFQueryExecutor(processor)
        logger.info("FastAPI application ready!")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        raise


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    # Get document statistics
    doc_stats = {}
    if processor:
        try:
            doc_stats = processor.get_document_statistics()
        except:
            doc_stats = {"total_documents": 0}
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "doc_stats": doc_stats
        }
    )


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Execute a document query"""
    if not executor:
        raise HTTPException(
            status_code=503,
            detail="Query system not initialized. Please check server logs."
        )
    
    try:
        logger.info(f"Received query: {request.query}")
        
        # Execute query
        result = executor.execute_query(request.query)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        logger.info(f"Query completed in {result['execution_time']:.2f}s")
        
        return QueryResponse(
            query=result['query'],
            answer=result['answer'],
            sources=result['sources'][:5],  # Limit to top 5 sources
            execution_time=result['execution_time'],
            multimodal_sources=result['multimodal_sources'],
            token_info=result['token_info'],
            cost_info=result['cost_info']
        )
        
    except Exception as e:
        logger.error(f"Query execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_statistics():
    """Get document processing statistics"""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        stats = processor.get_document_statistics()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if processor and executor else "initializing",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=3000,
        reload=True,
        log_level="info"
    )