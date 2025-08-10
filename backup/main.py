# main.py - HackRx Multilingual Document Query System
from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
import os
import asyncio
import time
import logging

from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool

# Import utility functions
from src.utils.document_loader import extract_text_from_document
from src.utils.text_splitter import split_text_into_chunks
from src.embeddings.embedding_model import EmbeddingModel
from src.vector_db.faiss_manager import FAISSManager
from src.llm.mistral_llm_client import MistralLLMClient
from src.cache.persistent_cache import persistent_cache
from src.utils.multilingual_query_handler import MultilingualQueryHandler
from src.utils.enhanced_query_processor import EnhancedQueryProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduced from INFO to WARNING
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep main app at INFO level

app = FastAPI(
    title="HackRx Multilingual Document Query System",
    description="An intelligent system for processing documents and answering multilingual queries.",
    version="2.0.0"
)

# --- Global components ---
startup_start_time = time.time()
embedding_generator = EmbeddingModel()
embedding_dimension = embedding_generator.get_embedding_dimension()
faiss_manager = FAISSManager(dimension=embedding_dimension)
mistral_llm_client = MistralLLMClient()
enhanced_processor = EnhancedQueryProcessor()
multilingual_handler = MultilingualQueryHandler(
    llm_client=mistral_llm_client,
    faiss_manager=faiss_manager,
    enhanced_processor=enhanced_processor,
    embedding_generator=embedding_generator
)
startup_end_time = time.time()

logger.info(f"🚀 Startup completed in {startup_end_time - startup_start_time:.2f}s")
logger.info(f"📊 Using embedding dimension: {embedding_dimension}")
logger.info("🌐 Multilingual query handler initialized with Mistral AI")

# --- Config ---
REQUIRED_AUTH_TOKEN = os.getenv("HACKRX_AUTH_TOKEN")

# --- Document Cache ---
document_cache: Dict[str, Tuple[List[str], List[List[float]]]] = {}
response_cache: Dict[str, Dict[str, Any]] = {}  # Cache for responses

# --- Pydantic Models ---
class RunRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# --- Authentication ---
async def verify_auth_token(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    token = auth_header.split(" ", 1)[1]
    if token != REQUIRED_AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# --- Main Endpoint ---
@app.post("/hackrx/run", response_model=RunResponse)
async def run_hackrx_submission(request: Request, payload: RunRequest):
    """
    Process multilingual document queries
    Supports Malayalam + English mixed content
    """
    await verify_auth_token(request)

    document_url = payload.documents
    questions = payload.questions
    
    logger.info(f"📋 Processing document: {document_url}")
    logger.info(f"❓ Questions: {len(questions)}")

    # Check response cache first
    cache_key = f"{document_url}:{hash(str(sorted(questions)))}"
    if cache_key in response_cache:
        logger.info("⚡ Using cached response")
        return RunResponse(answers=response_cache[cache_key])

    # Check if document is already processed and cached
    text_chunks, chunk_embeddings = None, None
    
    # First check persistent cache
    cached_data = persistent_cache.load_document(document_url)
    if cached_data:
        logger.info(f"📚 Using persistent cache for document")
        text_chunks, chunk_embeddings, metadata = cached_data
        logger.info(f"   - Chunks: {len(text_chunks)}")
        logger.info(f"   - Embeddings: {len(chunk_embeddings)} x {len(chunk_embeddings[0]) if chunk_embeddings else 0}")
    else:
        # Check in-memory cache as fallback
        if document_url in document_cache:
            logger.info(f"📋 Retrieving from in-memory cache")
            text_chunks, chunk_embeddings = document_cache[document_url]
        else:
            # Full document processing (slowest path)
            logger.info(f"⏳ Processing new document: {document_url}")
            
            # Step 1: Extract Text
            document_text = await run_in_threadpool(extract_text_from_document, document_url)
            if not document_text.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unable to extract text from document."
                )
            
            doc_length = len(document_text)
            logger.info(f"📄 Document length: {doc_length:,} characters")

            # Step 2: Chunking (optimized for multilingual content)
            CHUNK_SIZE = 2000
            CHUNK_OVERLAP = 150
            text_chunks = await run_in_threadpool(
                split_text_into_chunks, 
                document_text, 
                chunk_size=CHUNK_SIZE, 
                chunk_overlap=CHUNK_OVERLAP
            )
            logger.info(f"✂️ Document split into {len(text_chunks)} chunks")

            # Step 3: Embeddings
            logger.info(f"🔢 Generating embeddings for {len(text_chunks)} chunks...")
            chunk_embeddings = await run_in_threadpool(
                embedding_generator.get_embeddings, 
                text_chunks
            )
            if not chunk_embeddings:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Embedding generation failed."
                )
            
            logger.info(f"✅ Generated {len(chunk_embeddings)} embeddings")

            # Store in both caches for future use
            document_cache[document_url] = (text_chunks, chunk_embeddings)
            
            # Store in persistent cache
            try:
                persistent_cache.save_document(document_url, text_chunks, chunk_embeddings)
                logger.info("💾 Document cached for future use")
            except Exception as e:
                logger.warning(f"⚠️ Could not save to persistent cache: {e}")

    # Step 4: Populate FAISS Index
    current_faiss_docs = faiss_manager.get_total_documents()
    logger.info(f"🔍 Current FAISS documents: {current_faiss_docs}")
    
    # Always populate FAISS for the current request to ensure search works
    logger.info(f"📊 Adding {len(text_chunks)} chunks to FAISS index...")
    await run_in_threadpool(faiss_manager.add_documents, chunk_embeddings, text_chunks)
    
    new_faiss_total = faiss_manager.get_total_documents()
    logger.info(f"✅ FAISS index now contains {new_faiss_total} documents")

    # Step 5: Process multilingual queries
    logger.info("🌐 Processing queries with multilingual handler...")
    multilingual_result = await run_in_threadpool(
        multilingual_handler.process_multilingual_queries, 
        questions, 
        document_url
    )
    
    if not multilingual_result.get('success', False):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multilingual processing failed: {multilingual_result.get('error', 'Unknown error')}"
        )
    
    # Extract answers from results
    answers = []
    for result in multilingual_result.get('results', []):
        answers.append(result.get('answer', 'No answer generated'))
    
    # Log summary
    languages_detected = multilingual_result.get('languages_detected', [])
    categories_found = multilingual_result.get('categories_found', [])
    
    logger.info(f"✅ Processing complete:")
    logger.info(f"   - Total questions: {multilingual_result.get('total_questions', 0)}")
    logger.info(f"   - Languages detected: {', '.join(languages_detected)}")
    logger.info(f"   - Categories found: {', '.join(categories_found)}")
    
    # Cache the response
    response_cache[cache_key] = answers
    
    return RunResponse(answers=answers)

# --- Health Check ---
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "ok", 
        "message": "HackRx Multilingual System is running!",
        "languages_supported": ["English", "Malayalam"],
        "llm_provider": "Mistral AI"
    }

# --- Root endpoint ---
@app.get("/")
async def root():
    return {
        "message": "HackRx Multilingual Document Query System",
        "version": "2.0.0",
        "endpoint": "/hackrx/run",
        "languages": ["English", "Malayalam"],
        "authentication": "Bearer token required"
    }
