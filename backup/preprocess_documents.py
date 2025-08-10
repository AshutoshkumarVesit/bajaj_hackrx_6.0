# preprocess_documents.py
"""
Standalone script to preprocess documents and store them in persistent cache.
Run this script to preprocess known documents before the competition.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.utils.document_loader import extract_text_from_document
from src.utils.text_splitter import split_text_into_chunks
from src.embeddings.embedding_model import EmbeddingModel
from src.cache.persistent_cache import persistent_cache

async def preprocess_document(document_url: str, embedding_model: EmbeddingModel) -> bool:
    """
    Preprocess a single document and save to cache.
    
    Args:
        document_url: URL of the document to preprocess
        embedding_model: Initialized embedding model
    
    Returns:
        bool: True if successful
    """
    try:
        print(f"\n🔄 Processing: {document_url}")
        start_time = datetime.now()
        
        # Check if already cached
        if persistent_cache.is_cached(document_url):
            print(f"✅ Already cached, skipping: {document_url}")
            return True
        
        # Step 1: Extract text
        print("  📄 Extracting text...")
        document_text = extract_text_from_document(document_url)
        if not document_text.strip():
            print(f"❌ Failed to extract text from: {document_url}")
            return False
        
        doc_length = len(document_text)
        print(f"  📊 Document length: {doc_length:,} characters")
        print(f"  🎯 Processing entire document for maximum accuracy")
        
        # Step 2: Chunking
        print("  🔪 Chunking text...")
        CHUNK_SIZE = 2000
        CHUNK_OVERLAP = 150
        text_chunks = split_text_into_chunks(
            document_text, 
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        print(f"  📦 Created {len(text_chunks)} chunks")
        
        # Step 3: Generate embeddings
        print("  🧠 Generating embeddings...")
        chunk_embeddings = embedding_model.get_embeddings(text_chunks)
        if not chunk_embeddings:
            print(f"❌ Failed to generate embeddings for: {document_url}")
            return False
        
        print(f"  ✨ Generated {len(chunk_embeddings)} embeddings")
        
        # Step 4: Save to cache
        print("  💾 Saving to cache...")
        metadata = {
            'processed_at': datetime.now().isoformat(),
            'original_length': doc_length,
            'processed_length': len(document_text),
            'chunk_size': CHUNK_SIZE,
            'chunk_overlap': CHUNK_OVERLAP
        }
        
        success = persistent_cache.save_document(
            document_url, 
            text_chunks, 
            chunk_embeddings, 
            metadata
        )
        
        if success:
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ Completed in {processing_time:.2f}s: {document_url}")
            return True
        else:
            print(f"❌ Failed to cache: {document_url}")
            return False
        
    except Exception as e:
        print(f"❌ Error processing {document_url}: {e}")
        return False

async def preprocess_from_test_file(test_file_path: str = "queries_test.json"):
    """
    Preprocess all documents from the test file.
    
    Args:
        test_file_path: Path to the queries test file
    """
    try:
        # Load test data
        with open(test_file_path, 'r') as f:
            test_data = json.load(f)
        
        # Initialize embedding model
        print("🚀 Initializing embedding model...")
        embedding_model = EmbeddingModel()
        print(f"✅ Embedding model loaded (dimension: {embedding_model.get_embedding_dimension()})")
        
        # Extract unique document URLs
        document_urls = []
        for test_case in test_data:
            doc_url = test_case.get('documents')
            if doc_url and doc_url not in document_urls:
                document_urls.append(doc_url)
        
        print(f"\n📋 Found {len(document_urls)} unique documents to preprocess")
        
        # Process each document
        successful = 0
        failed = 0
        
        for i, doc_url in enumerate(document_urls, 1):
            print(f"\n{'='*60}")
            print(f"Processing document {i}/{len(document_urls)}")
            print(f"{'='*60}")
            
            success = await preprocess_document(doc_url, embedding_model)
            if success:
                successful += 1
            else:
                failed += 1
        
        # Summary
        print(f"\n{'='*60}")
        print(f"PREPROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Total: {len(document_urls)}")
        
        # Cache statistics
        print(f"\n📈 Cache Statistics:")
        stats = persistent_cache.get_cache_stats()
        print(f"  - Total documents in cache: {stats.get('total_documents', 0)}")
        print(f"  - Cache size: {stats.get('cache_size_mb', 0)} MB")
        
        return successful, failed
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return 0, 1

async def preprocess_single_document(document_url: str):
    """
    Preprocess a single document URL.
    
    Args:
        document_url: URL of the document to preprocess
    """
    print("🚀 Initializing embedding model...")
    embedding_model = EmbeddingModel()
    print(f"✅ Embedding model loaded (dimension: {embedding_model.get_embedding_dimension()})")
    
    success = await preprocess_document(document_url, embedding_model)
    return success

def show_cache_stats():
    """Show current cache statistics."""
    print("📈 Current Cache Statistics:")
    print("="*50)
    
    stats = persistent_cache.get_cache_stats()
    
    if 'error' in stats:
        print(f"❌ Error getting stats: {stats['error']}")
        return
    
    print(f"Total documents: {stats.get('total_documents', 0)}")
    print(f"Cache size: {stats.get('cache_size_mb', 0)} MB")
    print(f"Oldest cache: {stats.get('oldest_cache', 'N/A')}")
    print(f"Newest cache: {stats.get('newest_cache', 'N/A')}")
    
    print(f"\nDocument Details:")
    for doc in stats.get('documents', []):
        print(f"  - {doc.get('url', 'Unknown')[:80]}...")
        print(f"    Chunks: {doc.get('num_chunks', 0)}, Characters: {doc.get('total_characters', 0):,}")
        print(f"    Cached: {doc.get('cached_at', 'Unknown')}")

def clear_cache():
    """Clear the entire cache."""
    print("🗑️ Clearing entire cache...")
    success = persistent_cache.clear_cache()
    if success:
        print("✅ Cache cleared successfully")
    else:
        print("❌ Failed to clear cache")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess documents for persistent caching")
    parser.add_argument("--mode", choices=["all", "single", "stats", "clear"], 
                       default="all", help="Processing mode")
    parser.add_argument("--url", help="Document URL for single document processing")
    parser.add_argument("--file", default="queries_test.json", 
                       help="Test file path for batch processing")
    
    args = parser.parse_args()
    
    if args.mode == "all":
        print("🚀 Starting batch preprocessing from test file...")
        asyncio.run(preprocess_from_test_file(args.file))
    
    elif args.mode == "single":
        if not args.url:
            print("❌ Please provide --url for single document processing")
            sys.exit(1)
        print(f"🚀 Starting single document preprocessing: {args.url}")
        success = asyncio.run(preprocess_single_document(args.url))
        if success:
            print("✅ Document preprocessed successfully")
        else:
            print("❌ Document preprocessing failed")
    
    elif args.mode == "stats":
        show_cache_stats()
    
    elif args.mode == "clear":
        clear_cache()
