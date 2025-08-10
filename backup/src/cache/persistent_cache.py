# src/cache/persistent_cache.py
import os
import pickle
import hashlib
import json
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import asyncio
from pathlib import Path

class PersistentDocumentCache:
    """
    Persistent cache for preprocessed documents with embeddings.
    Stores chunks and embeddings to disk for instant retrieval.
    """
    
    def __init__(self, cache_dir: str = "document_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.chunks_dir = self.cache_dir / "chunks"
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.metadata_dir = self.cache_dir / "metadata"
        
        for dir_path in [self.chunks_dir, self.embeddings_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def _get_document_hash(self, document_url: str) -> str:
        """Generate a consistent hash for document URL."""
        return hashlib.md5(document_url.encode()).hexdigest()
    
    def _get_cache_paths(self, document_url: str) -> Dict[str, Path]:
        """Get file paths for cached document components."""
        doc_hash = self._get_document_hash(document_url)
        return {
            'chunks': self.chunks_dir / f"{doc_hash}_chunks.pkl",
            'embeddings': self.embeddings_dir / f"{doc_hash}_embeddings.pkl",
            'metadata': self.metadata_dir / f"{doc_hash}_metadata.json"
        }
    
    def is_cached(self, document_url: str) -> bool:
        """Check if document is already cached."""
        paths = self._get_cache_paths(document_url)
        return all(path.exists() for path in paths.values())
    
    def save_document(self, 
                     document_url: str, 
                     text_chunks: List[str], 
                     chunk_embeddings: List[List[float]],
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save processed document to persistent cache.
        
        Args:
            document_url: URL of the document
            text_chunks: List of text chunks
            chunk_embeddings: List of embedding vectors
            metadata: Optional metadata about the document
        
        Returns:
            bool: True if saved successfully
        """
        try:
            paths = self._get_cache_paths(document_url)
            
            # Save chunks
            with open(paths['chunks'], 'wb') as f:
                pickle.dump(text_chunks, f)
            
            # Save embeddings
            with open(paths['embeddings'], 'wb') as f:
                pickle.dump(chunk_embeddings, f)
            
            # Save metadata
            cache_metadata = {
                'document_url': document_url,
                'cached_at': datetime.now().isoformat(),
                'num_chunks': len(text_chunks),
                'embedding_dimension': len(chunk_embeddings[0]) if chunk_embeddings else 0,
                'total_characters': sum(len(chunk) for chunk in text_chunks),
                'custom_metadata': metadata or {}
            }
            
            with open(paths['metadata'], 'w') as f:
                json.dump(cache_metadata, f, indent=2)
            
            print(f"✅ Document cached successfully: {document_url}")
            print(f"   - Chunks: {len(text_chunks)}")
            print(f"   - Embeddings: {len(chunk_embeddings)}")
            return True
            
        except Exception as e:
            print(f"❌ Error caching document: {e}")
            return False
    
    def load_document(self, document_url: str) -> Optional[Tuple[List[str], List[List[float]], Dict[str, Any]]]:
        """
        Load processed document from persistent cache.
        
        Args:
            document_url: URL of the document
        
        Returns:
            Tuple of (text_chunks, chunk_embeddings, metadata) or None if not found
        """
        try:
            if not self.is_cached(document_url):
                return None
            
            paths = self._get_cache_paths(document_url)
            
            # Load chunks
            with open(paths['chunks'], 'rb') as f:
                text_chunks = pickle.load(f)
            
            # Load embeddings
            with open(paths['embeddings'], 'rb') as f:
                chunk_embeddings = pickle.load(f)
            
            # Load metadata
            with open(paths['metadata'], 'r') as f:
                metadata = json.load(f)
            
            print(f"✅ Document loaded from cache: {document_url}")
            print(f"   - Chunks: {len(text_chunks)}")
            print(f"   - Cached on: {metadata.get('cached_at', 'Unknown')}")
            
            return text_chunks, chunk_embeddings, metadata
            
        except Exception as e:
            print(f"❌ Error loading cached document: {e}")
            return None
    
    def clear_cache(self, document_url: Optional[str] = None) -> bool:
        """
        Clear cache for specific document or entire cache.
        
        Args:
            document_url: URL of specific document to clear, or None to clear all
        
        Returns:
            bool: True if cleared successfully
        """
        try:
            if document_url:
                # Clear specific document
                paths = self._get_cache_paths(document_url)
                for path in paths.values():
                    if path.exists():
                        path.unlink()
                print(f"✅ Cleared cache for: {document_url}")
            else:
                # Clear entire cache
                import shutil
                shutil.rmtree(self.cache_dir)
                self.__init__(str(self.cache_dir))
                print("✅ Cleared entire document cache")
            
            return True
            
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        try:
            stats = {
                'total_documents': len(list(self.metadata_dir.glob('*_metadata.json'))),
                'cache_size_mb': 0,
                'oldest_cache': None,
                'newest_cache': None,
                'documents': []
            }
            
            # Calculate total size
            for path in self.cache_dir.rglob('*'):
                if path.is_file():
                    stats['cache_size_mb'] += path.stat().st_size
            
            stats['cache_size_mb'] = round(stats['cache_size_mb'] / (1024 * 1024), 2)
            
            # Get document details
            for metadata_file in self.metadata_dir.glob('*_metadata.json'):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    cached_at = metadata.get('cached_at')
                    if cached_at:
                        if not stats['oldest_cache'] or cached_at < stats['oldest_cache']:
                            stats['oldest_cache'] = cached_at
                        if not stats['newest_cache'] or cached_at > stats['newest_cache']:
                            stats['newest_cache'] = cached_at
                    
                    stats['documents'].append({
                        'url': metadata.get('document_url', 'Unknown'),
                        'cached_at': cached_at,
                        'num_chunks': metadata.get('num_chunks', 0),
                        'total_characters': metadata.get('total_characters', 0)
                    })
                    
                except Exception as e:
                    print(f"Warning: Could not read metadata file {metadata_file}: {e}")
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting cache stats: {e}")
            return {'error': str(e)}

# Global cache instance
persistent_cache = PersistentDocumentCache()
